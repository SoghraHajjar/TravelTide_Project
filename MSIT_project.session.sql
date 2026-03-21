/* 
 This query fetches sum of page_clicks by # of seats
 booked for each year/month/day
 in sessions where the user booked a flight 
 */
SELECT date_part(
        'YEAR',
        to_timestamp(sessions.session_start, 'YYYY-MM-DD HH:MM:SS')
    ) AS year_,
    -- year to group by
    date_part(
        'MONTH',
        to_timestamp(sessions.session_start, 'YYYY-MM-DD HH:MM:SS')
    ) AS month_,
    -- month to group by
    date_part(
        'DAY',
        to_timestamp(sessions.session_start, 'YYYY-MM-DD HH:MM:SS')
    ) AS day_,
    -- day to group by
    flights.seats,
    -- seats
    -- sums page_clicks for each day
    SUM(sessions.page_clicks) AS total_clicks
FROM sessions -- joins on trip_id
    JOIN flights ON sessions.trip_id = flights.trip_id
WHERE -- filters for sessions with a booked flight
    sessions.flight_booked
GROUP BY YEAR(sessions.session_start, 'YYYY-MM-DD HH:MM:SS'),
    MONTH(sessions.session_start, 'YYYY-MM-DD HH:MM:SS'),
    DAY(sessions.session_start, 'YYYY-MM-DD HH:MM:SS'),
    flights.seats;