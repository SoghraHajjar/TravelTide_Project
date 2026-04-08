WITH active_users AS (
    SELECT user_id,
        COUNT(session_id) AS total_session
    FROM sessions
    WHERE session_start >= '2023-01-05'
    GROUP BY user_id
    HAVING COUNT(session_id) > 7
),
selected_sessions_trip_id AS (
    SELECT trip_id,
        user_id
    FROM sessions
    WHERE session_start >= '2023-01-05'
        AND user_id IN (
            SELECT user_id
            FROM active_users
        )
)
SELECT *
FROM users
WHERE user_id IN (
        SELECT user_id
        FROM selected_sessions_trip_id
    );
--SELECT *
--FROM hotels
--WHERE trip_id IN (
--SELECT trip_id
--FROM selected_sessions_trip_id
--);