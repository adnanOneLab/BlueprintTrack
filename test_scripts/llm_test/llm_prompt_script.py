import os
import json
import psycopg2
from psycopg2 import sql
from openai import OpenAI
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
from dataclasses import dataclass
import logging

# Configuration
DATABASE_CONFIG = {
    "host": "localhost",
    "database": "mall-user-details",
    "user" : "",
    "password" : "",
    "port": "5433"
}

OPENAI_API_KEY = "your_openai_api_key"
LLM_MODEL = "gpt-4"  # or "gpt-3.5-turbo" for cost savings

# Processing Configuration
@dataclass
class ProcessingConfig:
    max_users_to_process: int = 4
    max_visits_per_user: int = 20
    max_movements_per_user: int = 50
    days_lookback: int = 30
    output_format: str = "json"  # Options: "json", "csv", "both"
    save_to_file: bool = True
    output_directory: str = "insights_output"

class UserInsightGenerator:
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.setup_logging()
        
        try:
            self.db_conn = psycopg2.connect(**DATABASE_CONFIG)
            self.db_conn.autocommit = False
            self.logger.info("Database connection established.")
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            self.db_conn = None
            raise SystemExit(1)

        try:
            self.llm_client = OpenAI(api_key=OPENAI_API_KEY)
            self.logger.info("OpenAI client initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise SystemExit(1)
        
        # Create output directory
        if self.config.save_to_file:
            os.makedirs(self.config.output_directory, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('insight_generator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_users_to_process(self) -> List[str]:
        """Get list of users to process based on configuration"""
        with self.db_conn.cursor() as cursor:
            cursor.execute("""
                SELECT user_id FROM users 
                WHERE last_visit >= NOW() - INTERVAL '%s days'
                ORDER BY last_visit DESC
                LIMIT %s
            """, (self.config.days_lookback, self.config.max_users_to_process))
            return [row[0] for row in cursor.fetchall()]
    
    def _fetch_user_data(self, user_id: str) -> Dict[str, Any]:
        """Fetch comprehensive user data including visits, movements, and interests"""
        with self.db_conn.cursor() as cursor:
            # Get user basic info
            cursor.execute("""
                SELECT * FROM users WHERE user_id = %s
            """, (user_id,))
            user_row = cursor.fetchone()
            if not user_row:
                raise ValueError(f"User {user_id} not found")
            
            # Convert to dict using cursor description
            user_data = dict(zip([desc[0] for desc in cursor.description], user_row))
            
            # Get user visits (limited by config)
            cursor.execute("""
                SELECT * FROM visits 
                WHERE user_id = %s
                ORDER BY visit_date DESC
                LIMIT %s
            """, (user_id, self.config.max_visits_per_user))
            visits = []
            for row in cursor.fetchall():
                visits.append(dict(zip([desc[0] for desc in cursor.description], row)))
            user_data['visits'] = visits
            
            # Get visit movements for these visits (limited by config)
            visit_ids = [v['visit_id'] for v in visits]
            if visit_ids:
                cursor.execute("""
                    SELECT * FROM user_movements 
                    WHERE visit_id = ANY(%s)
                    ORDER BY start_time
                    LIMIT %s
                """, (visit_ids, self.config.max_movements_per_user))
                movements = []
                for row in cursor.fetchall():
                    movements.append(dict(zip([desc[0] for desc in cursor.description], row)))
                user_data['movements'] = movements
            else:
                user_data['movements'] = []
            
            # Get user interests
            cursor.execute("""
                SELECT i.name, ui.source 
                FROM user_interests ui
                JOIN interests i ON ui.interest_id = i.interest_id
                WHERE ui.user_id = %s
            """, (user_id,))
            interests = []
            for row in cursor.fetchall():
                interests.append(dict(zip([desc[0] for desc in cursor.description], row)))
            user_data['interests'] = interests
            
            return user_data
    
    def _fetch_store_data(self) -> Dict[str, Dict[str, str]]:
        """Fetch store data for reference"""
        with self.db_conn.cursor() as cursor:
            cursor.execute("""
                SELECT store_code, store_name, 
                       pattern_characterstic_1, pattern_characterstic_2, pattern_characterstic_3
                FROM stores
            """)
            stores = {}
            for row in cursor.fetchall():
                stores[row[0]] = {
                    'name': row[1],
                    'category': row[2] or 'Unknown',
                    'price_level': row[3] or 'Unknown',
                    'footfall_type': row[4] or 'Unknown'
                }
            return stores
    
    def _prepare_llm_prompt(self, user_data: Dict[str, Any], store_data: Dict[str, Dict[str, str]]) -> str:
        """Prepare a comprehensive prompt for the LLM"""
        # Simplify movements for the prompt
        simplified_movements = []
        for movement in user_data.get('movements', []):
            store_info = store_data.get(movement.get('store_code', ''), {})
            simplified_movements.append({
                'time': str(movement.get('start_time', '')),
                'activity': movement.get('situation', ''),
                'location': movement.get('location', ''),
                'store': store_info.get('name', ''),
                'store_category': store_info.get('category', '')
            })
        
        current_interests = ', '.join([i.get('name', '') for i in user_data.get('interests', [])]) or 'None'
        patterns = [user_data.get('pattern_1', ''), user_data.get('pattern_2', ''), user_data.get('pattern_3', '')]
        patterns_str = ', '.join([p for p in patterns if p]) or 'None'
        
        prompt = f"""
        Analyze the following user data to generate actionable marketing insights and recommendations.
        This user is part of a retail shopping mall's customer base.
        
        User Profile:
        - Name: {user_data.get('name', 'Unknown')}
        - Age: {self._calculate_age(user_data.get('date_of_birth')) if user_data.get('date_of_birth') else 'Unknown'}
        - Visit Frequency: {user_data.get('monthly_freq', 0)} days/month
        - Average Visit Duration: {user_data.get('avg_time_per_visit_year') or user_data.get('avg_time_per_visit_life', 'Unknown')}
        - Total Stores Visited: {user_data.get('stores_visited_life', 0)}
        - Current Interests: {current_interests}
        - Existing Behavior Patterns: {patterns_str}
        
        Recent Activity Analysis:
        - Number of recent visits analyzed: {len(user_data.get('visits', []))}
        - Number of store interactions: {len(simplified_movements)}
        
        Recent Store Interactions:
        {json.dumps(simplified_movements[:10], indent=2, default=str)}
        
        Based on this data, provide a comprehensive analysis with:
        
        1. **Customer Segment Classification**: What type of shopper is this person?
        2. **Interest Recommendations**: 3-5 new or refined interest categories
        3. **Store Recommendations**: 3-5 specific stores or store categories they should visit
        4. **Optimal Engagement Strategy**: Best times/methods for marketing outreach
        5. **Personalized Campaign Message**: A specific promotional message tailored to their behavior
        6. **Revenue Opportunity**: Estimated potential value and recommended spend targets
        7. **Behavioral Insights**: Key patterns that inform marketing strategy
        
        Provide your response in the following JSON structure:
        {{
            "user_id": "{user_data.get('user_id')}",
            "analysis_date": "{datetime.now().isoformat()}",
            "customer_segment": "string describing the customer type",
            "confidence_score": "number between 0-100 indicating confidence in analysis",
            "interest_recommendations": [
                {{"interest": "interest name", "reason": "why this interest fits"}}
            ],
            "store_recommendations": [
                {{"store_category": "category", "reason": "why recommend", "priority": "high/medium/low"}}
            ],
            "engagement_strategy": {{
                "optimal_contact_times": ["day/time recommendations"],
                "preferred_channels": ["email/sms/app notification/etc"],
                "frequency": "recommended contact frequency"
            }},
            "promotional_message": "personalized message text",
            "revenue_opportunity": {{
                "estimated_monthly_potential": "dollar amount or range",
                "recommended_offer_type": "discount/experience/bundle/etc",
                "target_categories": ["categories to focus promotions on"]
            }},
            "behavioral_insights": [
                "key insight 1",
                "key insight 2", 
                "key insight 3"
            ],
            "next_actions": [
                "specific actionable recommendation 1",
                "specific actionable recommendation 2"
            ]
        }}
        """
        return prompt
    
    def _calculate_age(self, dob: str) -> int:
        """Calculate age from date of birth"""
        try:
            birth_date = datetime.strptime(str(dob), '%Y-%m-%d').date()
            today = datetime.now().date()
            return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        except:
            return None
    
    def _get_llm_insights(self, prompt: str) -> Dict[str, Any]:
        """Get insights from LLM with error handling"""
        try:
            response = self.llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Return structured error response
            return {
                "error": "Failed to parse LLM response",
                "raw_response": response.choices[0].message.content if 'response' in locals() else "No response"
            }
        except Exception as e:
            self.logger.error(f"LLM API error: {e}")
            return {"error": str(e)}
    
    def _store_insights(self, user_id: str, insights: Dict[str, Any]) -> None:
        """Store the generated insights in the database"""
        if "error" in insights:
            self.logger.warning(f"Skipping database storage for user {user_id} due to insights error")
            return
            
        try:
            with self.db_conn.cursor() as cursor:
                # Create insights table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_insights (
                        insight_id SERIAL PRIMARY KEY,
                        user_id VARCHAR(255),
                        analysis_date TIMESTAMP DEFAULT NOW(),
                        customer_segment TEXT,
                        confidence_score INTEGER,
                        insights_json JSONB,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Store the complete insights
                cursor.execute("""
                    INSERT INTO user_insights (user_id, customer_segment, confidence_score, insights_json)
                    VALUES (%s, %s, %s, %s)
                """, (
                    user_id,
                    insights.get('customer_segment', ''),
                    insights.get('confidence_score', 0),
                    json.dumps(insights)
                ))
                
                # Update user patterns if we have behavioral insights
                behavioral_insights = insights.get('behavioral_insights', [])
                if len(behavioral_insights) >= 3:
                    cursor.execute("""
                        UPDATE users 
                        SET pattern_1 = %s, pattern_2 = %s, pattern_3 = %s
                        WHERE user_id = %s
                    """, (
                        behavioral_insights[0][:255],  # Limit length
                        behavioral_insights[1][:255],
                        behavioral_insights[2][:255],
                        user_id
                    ))
                
                # Store new interests
                for interest_rec in insights.get('interest_recommendations', []):
                    interest_name = interest_rec.get('interest', '')
                    if interest_name:
                        # Check if interest exists
                        cursor.execute("SELECT interest_id FROM interests WHERE name = %s", (interest_name,))
                        result = cursor.fetchone()
                        
                        if not result:
                            # Create new interest
                            cursor.execute("""
                                INSERT INTO interests (interest_id, name)
                                VALUES (uuid_generate_v4(), %s)
                                RETURNING interest_id
                            """, (interest_name,))
                            interest_id = cursor.fetchone()[0]
                        else:
                            interest_id = result[0]
                        
                        # Link to user
                        cursor.execute("""
                            INSERT INTO user_interests (user_id, interest_id, source, created_at)
                            VALUES (%s, %s, 'llm_inferred', NOW())
                            ON CONFLICT (user_id, interest_id) DO NOTHING
                        """, (user_id, interest_id))
                
                self.db_conn.commit()
                self.logger.info(f"Successfully stored insights for user {user_id}")
                
        except Exception as e:
            self.logger.error(f"Error storing insights for user {user_id}: {e}")
            self.db_conn.rollback()
    
    def _save_insights_to_file(self, all_insights: List[Dict[str, Any]]) -> None:
        """Save insights to file(s) based on configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config.output_format in ["json", "both"]:
            json_file = os.path.join(self.config.output_directory, f"insights_{timestamp}.json")
            with open(json_file, 'w') as f:
                json.dump(all_insights, f, indent=2, default=str)
            self.logger.info(f"Insights saved to {json_file}")
        
        if self.config.output_format in ["csv", "both"]:
            # Flatten the insights for CSV
            flattened_insights = []
            for insight in all_insights:
                if "error" not in insight:
                    flat_insight = {
                        'user_id': insight.get('user_id'),
                        'analysis_date': insight.get('analysis_date'),
                        'customer_segment': insight.get('customer_segment'),
                        'confidence_score': insight.get('confidence_score'),
                        'promotional_message': insight.get('promotional_message'),
                        'estimated_monthly_potential': insight.get('revenue_opportunity', {}).get('estimated_monthly_potential'),
                        'recommended_offer_type': insight.get('revenue_opportunity', {}).get('recommended_offer_type'),
                        'primary_interests': ', '.join([ir.get('interest', '') for ir in insight.get('interest_recommendations', [])[:3]]),
                        'top_store_recommendations': ', '.join([sr.get('store_category', '') for sr in insight.get('store_recommendations', [])[:3]]),
                        'key_behavioral_insights': ' | '.join(insight.get('behavioral_insights', [])[:3])
                    }
                    flattened_insights.append(flat_insight)
            
            if flattened_insights:
                df = pd.DataFrame(flattened_insights)
                csv_file = os.path.join(self.config.output_directory, f"insights_{timestamp}.csv")
                df.to_csv(csv_file, index=False)
                self.logger.info(f"Insights saved to {csv_file}")
    
    def generate_and_store_insights(self, user_id: str) -> Dict[str, Any]:
        """Main method to generate and store insights for a user"""
        try:
            self.logger.info(f"Processing user {user_id}...")
            
            # Fetch data
            user_data = self._fetch_user_data(user_id)
            store_data = self._fetch_store_data()
            
            # Prepare and send to LLM
            prompt = self._prepare_llm_prompt(user_data, store_data)
            insights = self._get_llm_insights(prompt)
            
            # Store results if no error
            if "error" not in insights:
                self._store_insights(user_id, insights)
            
            return insights
            
        except Exception as e:
            error_msg = f"Error processing user {user_id}: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg, "user_id": user_id}
    
    def process_all_users(self) -> List[Dict[str, Any]]:
        """Process all users and return comprehensive results"""
        user_ids = self.get_users_to_process()
        self.logger.info(f"Processing {len(user_ids)} users...")
        
        all_insights = []
        successful = 0
        failed = 0
        
        for i, user_id in enumerate(user_ids, 1):
            self.logger.info(f"Processing user {i}/{len(user_ids)}: {user_id}")
            
            insights = self.generate_and_store_insights(user_id)
            all_insights.append(insights)
            
            if "error" in insights:
                failed += 1
            else:
                successful += 1
            
            # Progress update every 10 users
            if i % 10 == 0:
                self.logger.info(f"Progress: {i}/{len(user_ids)} - Success: {successful}, Failed: {failed}")
        
        # Save to files if configured
        if self.config.save_to_file:
            self._save_insights_to_file(all_insights)
        
        # Summary
        self.logger.info(f"Processing complete: {successful} successful, {failed} failed")
        
        return all_insights
    
    def close(self):
        """Clean up resources"""
        if self.db_conn:
            self.db_conn.close()

# Example usage
if __name__ == "__main__":
    # Configure processing parameters
    config = ProcessingConfig(
        max_users_to_process=50,        # Process 50 users
        max_visits_per_user=15,         # Look at last 15 visits per user
        max_movements_per_user=30,      # Max 30 movements per user
        days_lookback=45,               # Look back 45 days
        output_format="both",           # Save as both JSON and CSV
        save_to_file=True,              # Save results to files
        output_directory="client_insights"  # Directory for output files
    )
    
    # Initialize and run
    generator = UserInsightGenerator(config)
    
    try:
        # Process all users
        all_insights = generator.process_all_users()
        
        # Print summary for immediate viewing
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        
        successful_insights = [i for i in all_insights if "error" not in i]
        failed_insights = [i for i in all_insights if "error" in i]
        
        print(f"Total Users Processed: {len(all_insights)}")
        print(f"Successful: {len(successful_insights)}")
        print(f"Failed: {len(failed_insights)}")
        
        if successful_insights:
            print(f"\nSample Insight for User {successful_insights[0].get('user_id')}:")
            print(f"Segment: {successful_insights[0].get('customer_segment')}")
            print(f"Confidence: {successful_insights[0].get('confidence_score')}%")
            print(f"Revenue Potential: {successful_insights[0].get('revenue_opportunity', {}).get('estimated_monthly_potential')}")
        
        print(f"\nResults saved to: {config.output_directory}/")
        
    finally:
        generator.close()