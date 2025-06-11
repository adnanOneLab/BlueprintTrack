-- Create the database (uncomment if you need to create the database)
-- CREATE DATABASE mall_analytics;
-- \c mall_analytics;

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create stores table (reference table that others depend on)
CREATE TABLE stores (
    store_code VARCHAR(10) PRIMARY KEY,
    store_name VARCHAR(100) NOT NULL,
    pattern_characterstic_1 VARCHAR(50),
    pattern_characterstic_2 VARCHAR(50),
    pattern_characterstic_3 VARCHAR(50)
);

-- Create interests table
CREATE TABLE interests (
    interest_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL
);

-- Create users table
CREATE TABLE users (
    user_id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100),
    date_of_birth DATE,
    address TEXT,
    cell_phone VARCHAR(20),
    picture_url VARCHAR(255),
    profiling_questions TEXT,
    
    -- Aggregated metrics
    monthly_visits INTEGER DEFAULT 0,
    yearly_visits INTEGER DEFAULT 0,
    life_visits INTEGER DEFAULT 0,
    
    avg_time_per_visit_year INTERVAL,
    avg_time_per_visit_life INTERVAL,
    
    stores_visited_month INTEGER DEFAULT 0,
    stores_visited_life INTEGER DEFAULT 0,
    
    first_visit DATE,
    last_visit DATE,
    recency INTEGER DEFAULT 0,
    monthly_freq INTEGER DEFAULT 0,
    
    -- Pattern fields
    pattern_1 VARCHAR(50),
    pattern_2 VARCHAR(50),
    pattern_3 VARCHAR(50)
);

-- Create visits table
CREATE TABLE visits (
    visit_id SERIAL PRIMARY KEY,
    user_id VARCHAR(20) REFERENCES users(user_id),
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    duration INTERVAL GENERATED ALWAYS AS (end_time - start_time) STORED,
    visit_date DATE GENERATED ALWAYS AS (start_time::DATE) STORED,
    stores_visited INTEGER DEFAULT 0
);

-- Create user_movements table
CREATE TABLE user_movements (
    movement_id SERIAL PRIMARY KEY,
    visit_id INTEGER REFERENCES visits(visit_id),
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    situation VARCHAR(50) NOT NULL CHECK (situation IN ('walking', 'standing', 'visit store', 'exit store')),
    camera_id VARCHAR(20) NOT NULL,
    location VARCHAR(100) NOT NULL,
    store_code VARCHAR(10) REFERENCES stores(store_code)
);

-- Create user_interests junction table
CREATE TABLE user_interests (
    user_id VARCHAR(20) REFERENCES users(user_id),
    interest_id UUID REFERENCES interests(interest_id),
    source VARCHAR(50) NOT NULL DEFAULT 'inferred',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, interest_id)
);

-- Insert dummy data into stores
INSERT INTO stores (store_code, store_name, pattern_characterstic_1, pattern_characterstic_2, pattern_characterstic_3) VALUES
('C101', 'Nike', 'Sportswear', 'Premium', 'High-traffic'),
('C102', 'Adidas', 'Sportswear', 'Mid-range', 'High-traffic'),
('C103', 'Zara', 'Fashion', 'Mid-range', 'High-traffic'),
('C104', 'H&M', 'Fashion', 'Budget', 'High-traffic'),
('C105', 'Apple Store', 'Electronics', 'Premium', 'Medium-traffic'),
('C106', 'Samsung', 'Electronics', 'Mid-range', 'Medium-traffic'),
('C107', 'Sephora', 'Beauty', 'Premium', 'High-traffic'),
('C108', 'MAC Cosmetics', 'Beauty', 'Premium', 'Medium-traffic'),
('C109', 'Starbucks', 'Food & Beverage', 'Mid-range', 'Very-high-traffic'),
('C110', 'Food Court', 'Food & Beverage', 'Budget', 'Very-high-traffic');

-- Insert dummy data into interests
INSERT INTO interests (interest_id, name) VALUES
(uuid_generate_v4(), 'Sportswear'),
(uuid_generate_v4(), 'Fashion'),
(uuid_generate_v4(), 'Electronics'),
(uuid_generate_v4(), 'Beauty Products'),
(uuid_generate_v4(), 'Luxury Goods'),
(uuid_generate_v4(), 'Budget Shopping'),
(uuid_generate_v4(), 'Coffee'),
(uuid_generate_v4(), 'Fast Food'),
(uuid_generate_v4(), 'Tech Gadgets'),
(uuid_generate_v4(), 'Skincare');

-- Insert dummy data into users
INSERT INTO users (user_id, name, email, date_of_birth, address, cell_phone, picture_url, 
                   monthly_visits, yearly_visits, life_visits, avg_time_per_visit_year, 
                   avg_time_per_visit_life, stores_visited_month, stores_visited_life,
                   first_visit, last_visit, recency, monthly_freq, pattern_1, pattern_2, pattern_3) VALUES
('A101XT20', 'John Doe', 'john.doe@example.com', '1985-07-15', '123 Main St, Cityville', '+15551234567', 'https://example.com/pics/john.jpg',
 4, 24, 56, '01:15:00', '01:05:00', 3, 12, '2022-01-10', CURRENT_DATE - INTERVAL '2 days', 2, 3, 'Explorer', 'Weekly', 'Sportswear Enthusiast'),

('B202YT30', 'Jane Smith', 'jane.smith@example.com', '1990-11-22', '456 Oak Ave, Townsville', '+15559876543', 'https://example.com/pics/jane.jpg',
 8, 62, 142, '00:45:00', '00:50:00', 5, 28, '2021-05-15', CURRENT_DATE - INTERVAL '1 day', 1, 6, 'Focused', 'Frequent', 'Beauty Shopper'),

('C303ZT40', 'Mike Johnson', 'mike.j@example.com', '1978-03-30', '789 Pine Rd, Villageton', '+15555551234', 'https://example.com/pics/mike.jpg',
 2, 12, 35, '02:30:00', '02:15:00', 2, 8, '2022-08-20', CURRENT_DATE - INTERVAL '5 days', 5, 2, 'Browser', 'Occasional', 'Tech Enthusiast'),

('D404WT50', 'Sarah Williams', 'sarah.w@example.com', '1995-09-05', '321 Elm Blvd, Hamletville', '+15553334455', 'https://example.com/pics/sarah.jpg',
 12, 98, 210, '00:30:00', '00:35:00', 7, 35, '2020-11-12', CURRENT_DATE, 0, 10, 'Fast Shopper', 'Daily', 'Fashion Lover');

-- Insert dummy visits for users
-- John Doe's visits
INSERT INTO visits (user_id, start_time, end_time, stores_visited) VALUES
('A101XT20', CURRENT_TIMESTAMP - INTERVAL '2 days' - INTERVAL '2 hours', CURRENT_TIMESTAMP - INTERVAL '2 days' - INTERVAL '45 minutes', 2),
('A101XT20', CURRENT_TIMESTAMP - INTERVAL '5 days' - INTERVAL '3 hours', CURRENT_TIMESTAMP - INTERVAL '5 days' - INTERVAL '1 hour', 1),
('A101XT20', CURRENT_TIMESTAMP - INTERVAL '12 days' - INTERVAL '1 hour', CURRENT_TIMESTAMP - INTERVAL '12 days' - INTERVAL '10 minutes', 3),
('A101XT20', CURRENT_TIMESTAMP - INTERVAL '20 days' - INTERVAL '4 hours', CURRENT_TIMESTAMP - INTERVAL '20 days' - INTERVAL '2 hours 30 minutes', 2),

-- Jane Smith's visits
('B202YT30', CURRENT_TIMESTAMP - INTERVAL '1 day' - INTERVAL '1 hour', CURRENT_TIMESTAMP - INTERVAL '1 day' - INTERVAL '20 minutes', 3),
('B202YT30', CURRENT_TIMESTAMP - INTERVAL '2 days' - INTERVAL '2 hours', CURRENT_TIMESTAMP - INTERVAL '2 days' - INTERVAL '1 hour', 2),
('B202YT30', CURRENT_TIMESTAMP - INTERVAL '4 days' - INTERVAL '3 hours', CURRENT_TIMESTAMP - INTERVAL '4 days' - INTERVAL '1 hour 15 minutes', 4),
('B202YT30', CURRENT_TIMESTAMP - INTERVAL '6 days' - INTERVAL '45 minutes', CURRENT_TIMESTAMP - INTERVAL '6 days' - INTERVAL '5 minutes', 1),
('B202YT30', CURRENT_TIMESTAMP - INTERVAL '8 days' - INTERVAL '2 hours 30 minutes', CURRENT_TIMESTAMP - INTERVAL '8 days' - INTERVAL '1 hour 45 minutes', 3),
('B202YT30', CURRENT_TIMESTAMP - INTERVAL '10 days' - INTERVAL '1 hour 15 minutes', CURRENT_TIMESTAMP - INTERVAL '10 days' - INTERVAL '30 minutes', 2),

-- Mike Johnson's visits
('C303ZT40', CURRENT_TIMESTAMP - INTERVAL '5 days' - INTERVAL '4 hours', CURRENT_TIMESTAMP - INTERVAL '5 days' - INTERVAL '1 hour 30 minutes', 1),
('C303ZT40', CURRENT_TIMESTAMP - INTERVAL '15 days' - INTERVAL '3 hours 15 minutes', CURRENT_TIMESTAMP - INTERVAL '15 days' - INTERVAL '45 minutes', 2),

-- Sarah Williams' visits
('D404WT50', CURRENT_TIMESTAMP - INTERVAL '1 hour', CURRENT_TIMESTAMP - INTERVAL '10 minutes', 3),
('D404WT50', CURRENT_TIMESTAMP - INTERVAL '1 day' - INTERVAL '2 hours', CURRENT_TIMESTAMP - INTERVAL '1 day' - INTERVAL '1 hour', 2),
('D404WT50', CURRENT_TIMESTAMP - INTERVAL '2 days' - INTERVAL '1 hour 30 minutes', CURRENT_TIMESTAMP - INTERVAL '2 days' - INTERVAL '45 minutes', 4),
('D404WT50', CURRENT_TIMESTAMP - INTERVAL '3 days' - INTERVAL '3 hours', CURRENT_TIMESTAMP - INTERVAL '3 days' - INTERVAL '1 hour 15 minutes', 3),
('D404WT50', CURRENT_TIMESTAMP - INTERVAL '4 days' - INTERVAL '2 hours', CURRENT_TIMESTAMP - INTERVAL '4 days' - INTERVAL '1 hour', 2),
('D404WT50', CURRENT_TIMESTAMP - INTERVAL '5 days' - INTERVAL '1 hour 45 minutes', CURRENT_TIMESTAMP - INTERVAL '5 days' - INTERVAL '45 minutes', 1);

-- Insert user movements for visits
-- For John Doe's first visit
INSERT INTO user_movements (visit_id, start_time, end_time, situation, camera_id, location, store_code) VALUES
(1, (SELECT start_time FROM visits WHERE visit_id = 1), (SELECT start_time FROM visits WHERE visit_id = 1) + INTERVAL '15 minutes', 'walking', 'CAM101', 'North Corridor', NULL),
(1, (SELECT start_time FROM visits WHERE visit_id = 1) + INTERVAL '15 minutes', (SELECT start_time FROM visits WHERE visit_id = 1) + INTERVAL '25 minutes', 'visit store', 'CAM102', 'Store Entrance', 'C101'),
(1, (SELECT start_time FROM visits WHERE visit_id = 1) + INTERVAL '25 minutes', (SELECT start_time FROM visits WHERE visit_id = 1) + INTERVAL '55 minutes', 'standing', 'CAM103', 'Nike Store', 'C101'),
(1, (SELECT start_time FROM visits WHERE visit_id = 1) + INTERVAL '55 minutes', (SELECT start_time FROM visits WHERE visit_id = 1) + INTERVAL '65 minutes', 'exit store', 'CAM102', 'Store Exit', 'C101'),
(1, (SELECT start_time FROM visits WHERE visit_id = 1) + INTERVAL '65 minutes', (SELECT start_time FROM visits WHERE visit_id = 1) + INTERVAL '75 minutes', 'walking', 'CAM104', 'Central Atrium', NULL);

-- For Jane Smith's first visit
INSERT INTO user_movements (visit_id, start_time, end_time, situation, camera_id, location, store_code) VALUES
(5, (SELECT start_time FROM visits WHERE visit_id = 5), (SELECT start_time FROM visits WHERE visit_id = 5) + INTERVAL '10 minutes', 'walking', 'CAM201', 'East Wing', NULL),
(5, (SELECT start_time FROM visits WHERE visit_id = 5) + INTERVAL '10 minutes', (SELECT start_time FROM visits WHERE visit_id = 5) + INTERVAL '15 minutes', 'visit store', 'CAM202', 'Beauty Section', 'C107'),
(5, (SELECT start_time FROM visits WHERE visit_id = 5) + INTERVAL '15 minutes', (SELECT start_time FROM visits WHERE visit_id = 5) + INTERVAL '35 minutes', 'standing', 'CAM203', 'Sephora', 'C107'),
(5, (SELECT start_time FROM visits WHERE visit_id = 5) + INTERVAL '35 minutes', (SELECT start_time FROM visits WHERE visit_id = 5) + INTERVAL '40 minutes', 'exit store', 'CAM202', 'Beauty Section', 'C107');

-- For Mike Johnson's first visit
INSERT INTO user_movements (visit_id, start_time, end_time, situation, camera_id, location, store_code) VALUES
(11, (SELECT start_time FROM visits WHERE visit_id = 11), (SELECT start_time FROM visits WHERE visit_id = 11) + INTERVAL '5 minutes', 'walking', 'CAM301', 'Tech Zone', NULL),
(11, (SELECT start_time FROM visits WHERE visit_id = 11) + INTERVAL '5 minutes', (SELECT start_time FROM visits WHERE visit_id = 11) + INTERVAL '10 minutes', 'visit store', 'CAM302', 'Apple Store Entrance', 'C105'),
(11, (SELECT start_time FROM visits WHERE visit_id = 11) + INTERVAL '10 minutes', (SELECT start_time FROM visits WHERE visit_id = 11) + INTERVAL '2 hours', 'standing', 'CAM303', 'Apple Store Demo Area', 'C105'),
(11, (SELECT start_time FROM visits WHERE visit_id = 11) + INTERVAL '2 hours', (SELECT start_time FROM visits WHERE visit_id = 11) + INTERVAL '2 hours 10 minutes', 'exit store', 'CAM302', 'Apple Store Exit', 'C105'),
(11, (SELECT start_time FROM visits WHERE visit_id = 11) + INTERVAL '2 hours 10 minutes', (SELECT start_time FROM visits WHERE visit_id = 11) + INTERVAL '2 hours 30 minutes', 'visit store', 'CAM304', 'Samsung Entrance', 'C106'),
(11, (SELECT start_time FROM visits WHERE visit_id = 11) + INTERVAL '2 hours 30 minutes', (SELECT end_time FROM visits WHERE visit_id = 11), 'standing', 'CAM305', 'Samsung TV Section', 'C106');

-- Insert user interests
INSERT INTO user_interests (user_id, interest_id, source) VALUES
-- John Doe's interests
('A101XT20', (SELECT interest_id FROM interests WHERE name = 'Sportswear'), 'inferred'),
('A101XT20', (SELECT interest_id FROM interests WHERE name = 'Tech Gadgets'), 'manual'),

-- Jane Smith's interests
('B202YT30', (SELECT interest_id FROM interests WHERE name = 'Beauty Products'), 'inferred'),
('B202YT30', (SELECT interest_id FROM interests WHERE name = 'Luxury Goods'), 'inferred'),
('B202YT30', (SELECT interest_id FROM interests WHERE name = 'Fashion'), 'manual'),

-- Mike Johnson's interests
('C303ZT40', (SELECT interest_id FROM interests WHERE name = 'Electronics'), 'inferred'),
('C303ZT40', (SELECT interest_id FROM interests WHERE name = 'Tech Gadgets'), 'inferred'),

-- Sarah Williams' interests
('D404WT50', (SELECT interest_id FROM interests WHERE name = 'Fashion'), 'inferred'),
('D404WT50', (SELECT interest_id FROM interests WHERE name = 'Coffee'), 'manual'),
('D404WT50', (SELECT interest_id FROM interests WHERE name = 'Beauty Products'), 'inferred');

-- Update aggregated fields in users table based on the inserted data
UPDATE users u SET
    first_visit = subq.first_visit,
    last_visit = subq.last_visit,
    life_visits = subq.visit_count,
    stores_visited_life = subq.stores_visited
FROM (
    SELECT 
        user_id, 
        MIN(visit_date) AS first_visit, 
        MAX(visit_date) AS last_visit,
        COUNT(*) AS visit_count,
        SUM(stores_visited) AS stores_visited
    FROM visits
    GROUP BY user_id
) AS subq
WHERE u.user_id = subq.user_id;

-- Update recency for all users
UPDATE users SET recency = CURRENT_DATE - last_visit;

-- View some data to verify
SELECT * FROM users;
SELECT * FROM visits LIMIT 5;
SELECT * FROM user_movements LIMIT 5;
SELECT * FROM user_interests;