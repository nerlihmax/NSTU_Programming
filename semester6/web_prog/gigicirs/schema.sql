DROP SCHEMA IF EXISTS public CASCADE;
CREATE SCHEMA IF NOT EXISTS public;

CREATE TABLE author
(
    id      serial PRIMARY KEY,
    name    text NOT NULL,
    address text NOT NULL
);

CREATE TABLE publisher
(
    id      serial PRIMARY KEY,
    name    text NOT NULL,
    address text NOT NULL
);

CREATE TABLE seller
(
    id   serial PRIMARY KEY,
    name text NOT NULL
);

CREATE TABLE book
(
    id           serial PRIMARY KEY,
    title        text NOT NULL,
    author_id    int REFERENCES author (id),
    publisher_id int REFERENCES publisher (id),
    year         int  NOT NULL,
    price        real NOT NULL,
    seller_id    int REFERENCES seller (id)
);

INSERT INTO author (name, address)
VALUES ('J. K. Rowling', 'London, UK'),
       ('J. R. R. Tolkien', 'Bournmouth, UK'),
       ('Stephen King', 'Portland, ME, USA'),
       ('George R. R. Martin', 'Bayonne, NJ, USA'),
       ('Dan Brown', 'Exeter, NH, USA'),
       ('J. D. Salinger', 'New York, NY, USA'),
       ('J. D. Robb', 'New York, NY, USA');

INSERT INTO publisher (name, address)
VALUES ('Bloomsbury', 'London, UK'),
       ('Allen & Unwin', 'Sydney, Australia'),
       ('Houghton Mifflin', 'Boston, MA, USA'),
       ('HarperCollins', 'New York, NY, USA'),
       ('Bloomsbury', 'London, UK');

INSERT INTO seller (name)
VALUES ('Amazon'),
       ('Barnes & Noble'),
       ('Books-A-Million');

INSERT INTO book (title, author_id, publisher_id, year, price, seller_id)
VALUES ('Harry Potter and the Philosopher''s Stone', 1, 1, 1997, 8.99, 1),
       ('Harry Potter and the Chamber of Secrets', 1, 1, 1998, 9.50, 1),
       ('Harry Potter and the Prisoner of Azkaban', 1, 1, 1999, 5.20, 1),
       ('Harry Potter and the Goblet of Fire', 1, 1, 2000, 4.40, 1),
       ('Harry Potter and the Order of the Phoenix', 1, 1, 2003, 8.99, 1),
       ('Harry Potter and the Half-Blood Prince', 1, 1, 2005, 9.10, 1),
       ('Harry Potter and the Deathly Hallows', 1, 1, 2007, 8.16, 1),
       ('The Fellowship of the Ring', 2, 2, 1954, 8.99, 2),
       ('The Two Towers', 2, 2, 1954, 2.50, 2),
       ('The Return of the King', 2, 2, 1955, 28.99, 2),
       ('The Shining', 3, 3, 1977, 31.99, 3),
       ('The Stand', 3, 3, 1978, 12.99, 3),
       ('It', 3, 3, 1986, 21.15, 3),
       ('The Gunslinger', 4, 4, 1982, 1.99, 1),
       ('The Drawing of the Three', 4, 4, 1987, 18.99, 1),
       ('The Waste Lands', 4, 4, 1991, 16.99, 2),
       ('The Dark Tower', 4, 4, 2003, 6.99, 2),
       ('The Da Vinci Code', 5, 5, 2003, 7.99, 1),
       ('Angels & Demons', 5, 5, 2000, 100, 1),
       ('The Lost Symbol', 5, 5, 2009, 20, 2),
       ('The Catcher in the Rye', 6, 1, 1951, 8, 3),
       ('The Hobbit', 7, 2, 1937, 7, 2),
       ('The Silmarillion', 7, 3, 1977, 12, 1),
       ('A Game of Thrones', 1, 4, 1996, 17, 2),
       ('A Clash of Kings', 2, 5, 1998, 18, 3),
       ('A Storm of Swords', 3, 1, 2000, 17.99, 1),
       ('A Feast for Crows', 4, 2, 2005, 60.99, 2),
       ('A Dance with Dragons', 5, 4, 2011, 11.99, 1);

CREATE TABLE users
(
    id           serial PRIMARY KEY,
    login        text NOT NULL,
    password     text NOT NULL,
    access_level int  NOT NULL DEFAULT 0
);

INSERT INTO users (login, password, access_level)
VALUES ('admin', 'adminpassword', 2),
       ('operator', 'operatorpassword', 1);

SELECT book.id,
       book.title,
       book.year,
       book.price,
       a.name    AS author,
       p.name    AS publisher,
       s.name    AS seller,
       a.address AS author_address,
       p.address AS publisher_address
FROM book
         INNER JOIN author a ON a.id = book.author_id
         INNER JOIN publisher p ON book.publisher_id = p.id
         INNER JOIN seller s ON s.id = book.seller_id;