import os
import json
import psycopg2
from psycopg2 import sql
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
from dataclasses import dataclass
import logging
import requests
from collections import Counter

from dotenv import load_dotenv  


load_dotenv()
# Get environment variables
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_DB_NAME = os.getenv('POSTGRES_DB_NAME', None)
POSTGRES_USER=os.getenv('POSTGRES_USER', "postgres")
POSTGRES_PASSWORD=os.getenv('POSTGRES_PASSWORD', None)
POSTGRES_PORT=os.getenv('POSTGRES_PORT', 5433)

GROK_API_KEY=os.getenv("GROK_API_KEY", None)

# Configuration
DATABASE_CONFIG = {
    "host": POSTGRES_HOST,
    "database": POSTGRES_DB_NAME,
    "user" : POSTGRES_USER,
    "password" : POSTGRES_PASSWORD,
    "port": POSTGRES_PORT
}

# Free LLM Options Configuration
LLM_CONFIG = {
    "provider": "groq",
    # Groq (free tier available)
    "groq": {
        "api_key": GROK_API_KEY,  # Get free from groq.com
        "model": "llama3-8b-8192",
        "base_url": "https://api.groq.com/openai/v1"
    }
}

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

class FreeLLMClient:
    """Wrapper for different free LLM providers"""
    
    def __init__(self, provider: str, config: dict):
        self.provider = provider
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def chat_completion(self, messages: List[Dict], temperature: float = 0.7) -> str:
        """Get chat completion from the configured LLM provider"""
        try:
            return self._groq_completion(messages, temperature)
        except Exception as e:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _groq_completion(self, messages: List[Dict], temperature: float) -> str:
        """Groq completion (OpenAI-compatible API)"""
        try:
            headers = {
                "Authorization": f"Bearer {self.config['api_key']}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.config['base_url']}/chat/completions",
                headers=headers,
                json={
                    "model": self.config["model"],
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 2000
                },
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            self.logger.error(f"Groq API error: {e}")
            raise

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

        # Initialize free LLM client
        try:
            provider = LLM_CONFIG["provider"]
            provider_config = LLM_CONFIG[provider]
            self.llm_client = FreeLLMClient(provider, provider_config)
            self.logger.info(f"Free LLM client initialized with provider: {provider}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
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
                logging.FileHandler('test_scripts/llm_test/insight_generator.log'),
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
    
    def _analyze_store_visits(self, user_data: Dict[str, Any], store_data: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        """Analyze user's store visit patterns"""
        store_visits = {}
        store_categories = Counter()
        store_time_spent = {}
        visit_frequency = Counter()
        
        # Analyze movements to extract store visit patterns
        for movement in user_data.get('movements', []):
            store_code = movement.get('store_code')
            situation = movement.get('situation', '')
            
            if store_code and situation in ['visit store', 'exit store']:
                store_info = store_data.get(store_code, {})
                store_name = store_info.get('name', store_code)
                store_category = store_info.get('category', 'Unknown')
                
                # Count visits per store
                if store_name not in store_visits:
                    store_visits[store_name] = {
                        'visits': 0,
                        'category': store_category,
                        'price_level': store_info.get('price_level', 'Unknown'),
                        'footfall_type': store_info.get('footfall_type', 'Unknown'),
                        'total_time': timedelta(0),
                        'avg_time_per_visit': None
                    }
                
                if situation == 'visit store':
                    store_visits[store_name]['visits'] += 1
                    visit_frequency[store_name] += 1
                    store_categories[store_category] += 1
                    
                    # Calculate time spent (if we have exit time)
                    start_time = movement.get('start_time')
                    end_time = movement.get('end_time')
                    if start_time and end_time:
                        time_spent = end_time - start_time
                        if store_name not in store_time_spent:
                            store_time_spent[store_name] = []
                        store_time_spent[store_name].append(time_spent)
        
        # Calculate average time spent per store
        for store_name in store_time_spent:
            times = store_time_spent[store_name]
            if times:
                avg_time = sum(times, timedelta(0)) / len(times)
                store_visits[store_name]['avg_time_per_visit'] = str(avg_time)
                store_visits[store_name]['total_time'] = sum(times, timedelta(0))
        
        # Sort stores by visit frequency
        sorted_stores = sorted(store_visits.items(), key=lambda x: x[1]['visits'], reverse=True)
        
        # Get top categories
        top_categories = store_categories.most_common(5)
        
        return {
            'total_unique_stores': len(store_visits),
            'total_store_visits': sum(visit_frequency.values()),
            'top_stores': dict(sorted_stores[:10]),  # Top 10 most visited stores
            'favorite_categories': [{'category': cat, 'visits': count} for cat, count in top_categories],
            'store_loyalty_score': self._calculate_loyalty_score(visit_frequency),
            'category_diversity_score': len(store_categories) / max(len(store_visits), 1) if store_visits else 0
        }
    
    def _calculate_loyalty_score(self, visit_frequency: Counter) -> float:
        """Calculate store loyalty score (0-1, higher = more loyal to specific stores)"""
        if not visit_frequency:
            return 0.0
        
        total_visits = sum(visit_frequency.values())
        if total_visits == 0:
            return 0.0
        
        # Calculate concentration - if user visits few stores frequently, loyalty is high
        # If user visits many stores infrequently, loyalty is low
        unique_stores = len(visit_frequency)
        if unique_stores <= 1:
            return 1.0
        
        # Calculate Herfindahl index (concentration measure)
        proportions = [count / total_visits for count in visit_frequency.values()]
        herfindahl_index = sum(p ** 2 for p in proportions)
        
        # Normalize to 0-1 scale
        min_herfindahl = 1 / unique_stores  # Perfect distribution
        max_herfindahl = 1.0  # All visits to one store
        
        if max_herfindahl == min_herfindahl:
            return 1.0
        
        loyalty_score = (herfindahl_index - min_herfindahl) / (max_herfindahl - min_herfindahl)
        return round(loyalty_score, 3)
    
    def _prepare_llm_prompt(self, user_data: Dict[str, Any], store_data: Dict[str, Dict[str, str]]) -> str:
        """Prepare a comprehensive prompt for the LLM with detailed store visit analysis"""
        # Analyze store visits
        store_analysis = self._analyze_store_visits(user_data, store_data)
        
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
        
        # Format top stores for the prompt
        top_stores_info = []
        for store_name, info in list(store_analysis['top_stores'].items())[:5]:
            avg_time = info.get('avg_time_per_visit', 'Unknown')
            top_stores_info.append(f"- {store_name} ({info['category']}): {info['visits']} visits, avg time: {avg_time}")
        
        top_stores_str = '\n'.join(top_stores_info) if top_stores_info else 'None'
        
        # Format favorite categories
        fav_categories_str = ', '.join([f"{cat['category']} ({cat['visits']} visits)" 
                                       for cat in store_analysis['favorite_categories'][:3]]) or 'None'
        
        prompt = f"""
        You are a retail analytics expert. Analyze the following user data to generate marketing insights.
        
        User Profile:
        - Name: {user_data.get('name', 'Unknown')}
        - Age: {self._calculate_age(user_data.get('date_of_birth')) if user_data.get('date_of_birth') else 'Unknown'}
        - Visit Frequency: {user_data.get('monthly_freq', 0)} days/month
        - Average Visit Duration: {user_data.get('avg_time_per_visit_year') or user_data.get('avg_time_per_visit_life', 'Unknown')}
        - Total Stores Visited: {user_data.get('stores_visited_life', 0)}
        - Current Interests: {current_interests}
        - Existing Behavior Patterns: {patterns_str}
        
        Store Visit Analysis:
        - Total Unique Stores Visited: {store_analysis['total_unique_stores']}
        - Total Store Visits: {store_analysis['total_store_visits']}
        - Store Loyalty Score: {store_analysis['store_loyalty_score']} (0=explorer, 1=loyal to few stores)
        - Category Diversity Score: {store_analysis['category_diversity_score']:.2f}
        
        Most Visited Stores:
{top_stores_str}
        
        Favorite Store Categories: {fav_categories_str}
        
        Recent Activity:
        - Recent visits: {len(user_data.get('visits', []))}
        - Store interactions: {len(simplified_movements)}
        
        Recent Store Interactions (first 5):
        {json.dumps(simplified_movements[:5], indent=2, default=str)}
        
        Based on this comprehensive store visit data, provide analysis in this exact JSON format (no extra text):
        {{
            "user_id": "{user_data.get('user_id')}",
            "analysis_date": "{datetime.now().isoformat()}",
            "customer_segment": "brief description of customer type based on store preferences",
            "confidence_score": 85,
            "store_visit_insights": {{
                "shopping_style": "focused/explorer/mixed based on loyalty and diversity scores",
                "preferred_categories": [list of top 3 categories based on visits],
                "loyalty_level": "high/medium/low",
                "price_sensitivity": "based on visited store price levels",
                "visit_pattern": "frequent short visits / infrequent long visits / etc"
            }},
            "interest_recommendations": [
                {{"interest": "Fashion", "reason": "visits clothing stores frequently", "confidence": "high"}},
                {{"interest": "Technology", "reason": "spends time in electronics section", "confidence": "medium"}}
            ],
            "store_recommendations": [
                {{"store_category": "Premium Fashion", "reason": "matches current shopping pattern at similar stores", "priority": "high"}},
                {{"store_category": "Home Decor", "reason": "complements current lifestyle interests", "priority": "medium"}}
            ],
            "cross_selling_opportunities": [
                {{"from_category": "Fashion", "to_category": "Accessories", "reason": "natural complement to current shopping behavior"}},
                {{"from_category": "Electronics", "to_category": "Tech Accessories", "reason": "frequently purchased together"}}
            ],
            "engagement_strategy": {{
                "optimal_contact_times": ["weekend afternoons based on visit patterns"],
                "preferred_channels": ["email", "app notification"],
                "frequency": "weekly",
                "personalization_focus": "store-specific offers vs category-wide promotions"
            }},
            "promotional_message": "Personalized offer message referencing their favorite stores/categories",
            "revenue_opportunity": {{
                "estimated_monthly_potential": "$150-200",
                "recommended_offer_type": "store-specific discount vs category discount",
                "target_categories": ["fashion", "accessories"],
                "upselling_potential": "high/medium/low based on current spending pattern"
            }},
            "behavioral_insights": [
                "Store loyalty insights",
                "Shopping frequency patterns",
                "Category preferences",
                "Time spending behavior"
            ],
            "next_actions": [
                "Send targeted offers for favorite store categories",
                "Introduce similar stores in preferred categories",
                "Create loyalty program for frequently visited stores"
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
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.chat_completion(messages, temperature=0.7)
            
            # Try to extract JSON from response
            # Sometimes LLMs add extra text, so we look for JSON block
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                # Try parsing the entire response
                return json.loads(response_text)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Return structured error response with the raw response for debugging
            return {
                "error": "Failed to parse LLM response",
                "raw_response": response_text if 'response_text' in locals() else "No response",
                "user_id": prompt.split('"user_id": "')[1].split('"')[0] if '"user_id":' in prompt else "unknown"
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
            json_file = os.path.join(self.config.output_directory, f"customer_insights_{timestamp}.json")
            with open(json_file, 'w') as f:
                json.dump(all_insights, f, indent=2, default=str)
            self.logger.info(f"Insights saved to {json_file}")
        
        if self.config.output_format in ["csv", "both"]:
            # Flatten the insights for CSV
            flattened_insights = []
            for insight in all_insights:
                if "error" not in insight:
                    store_insights = insight.get('store_visit_insights', {})
                    flat_insight = {
                        'user_id': insight.get('user_id'),
                        'analysis_date': insight.get('analysis_date'),
                        'customer_segment': insight.get('customer_segment'),
                        'confidence_score': insight.get('confidence_score'),
                        'shopping_style': store_insights.get('shopping_style'),
                        'loyalty_level': store_insights.get('loyalty_level'),
                        'price_sensitivity': store_insights.get('price_sensitivity'),
                        'preferred_categories': ', '.join(store_insights.get('preferred_categories', [])),
                        'promotional_message': insight.get('promotional_message'),
                        'estimated_monthly_potential': insight.get('revenue_opportunity', {}).get('estimated_monthly_potential'),
                        'recommended_offer_type': insight.get('revenue_opportunity', {}).get('recommended_offer_type'),
                        'primary_interests': ', '.join([ir.get('interest', '') for ir in insight.get('interest_recommendations', [])[:3]]),
                        'top_store_recommendations': ', '.join([sr.get('store_category', '') for sr in insight.get('store_recommendations', [])[:3]]),
                        'key_behavioral_insights': ' | '.join(insight.get('behavioral_insights', [])[:3]),
                        'cross_selling_opportunities': ' | '.join([f"{cs.get('from_category', '')} -> {cs.get('to_category', '')}" 
                                                                  for cs in insight.get('cross_selling_opportunities', [])[:2]])
                    }
                    flattened_insights.append(flat_insight)
            
            if flattened_insights:
                df = pd.DataFrame(flattened_insights)
                csv_file = os.path.join(self.config.output_directory, f"customer_insights_{timestamp}.csv")
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
        max_users_to_process=5,         # Start small for testing
        max_visits_per_user=10,         # Reduce for faster processing
        max_movements_per_user=20,      # Reduce for faster processing
        days_lookback=30,               
        output_format="both",           
        save_to_file=True,              
        output_directory="test_scripts/llm_test/client_insights"  
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
            for insights in successful_insights:
                print(f"\nSample Insight for User {insights.get('user_id')}:")
                print(f"Segment: {insights.get('customer_segment')}")
                print(f"Shopping Style: {insights.get('store_visit_insights', {}).get('shopping_style')}")
                print(f"Loyalty Level: {insights.get('store_visit_insights', {}).get('loyalty_level')}")
                print(f"Confidence: {insights.get('confidence_score')}%")
                print(f"Revenue Potential: {insights.get('revenue_opportunity', {}).get('estimated_monthly_potential')}")
                print(f"Preferred Categories: {', '.join(insights.get('store_visit_insights', {}).get('preferred_categories', []))}")
        
        if failed_insights:
            print(f"\nFirst few errors:")
            for i, failed in enumerate(failed_insights[:3]):
                print(f"Error {i+1}: {failed.get('error', 'Unknown error')}")
        
        print(f"\nResults saved to: {config.output_directory}/")
        
    finally:
        generator.close()