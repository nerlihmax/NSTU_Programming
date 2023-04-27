CREATE TABLE IF NOT EXISTS my_table (
    id SERIAL NOT NULL CONSTRAINT pkey PRIMARY KEY,
    title TEXT,
    likes INT NOT NULL DEFAULT 0
);


INSERT INTO my_table (title, likes)
VALUES (
    'Titanic',
    99
),
(
    'Green mile',
    1234
),
(
    'Brother',
    987654321
);