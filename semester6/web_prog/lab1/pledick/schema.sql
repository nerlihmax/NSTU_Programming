drop schema if exists public cascade;
create schema if not exists public;

create table author
(
    id      serial primary key,
    name    text not null,
    address text not null
);

create table publisher
(
    id      serial primary key,
    name    text not null,
    address text not null
);

create table seller
(
    id   serial primary key,
    name text not null
);

create table book
(
    id           serial primary key,
    title        text not null,
    author_id    int references author (id),
    publisher_id int references publisher (id),
    year         int  not null,
    price        real not null,
    seller_id    int references seller (id)
);

insert into author (name, address)
values ('J. K. Rowling', 'London, UK'),
       ('J. R. R. Tolkien', 'Bournmouth, UK'),
       ('Stephen King', 'Portland, ME, USA'),
       ('George R. R. Martin', 'Bayonne, NJ, USA'),
       ('Dan Brown', 'Exeter, NH, USA'),
       ('J. D. Salinger', 'New York, NY, USA'),
       ('J. R. R. Martin', 'Bayonne, NJ, USA'),
       ('J. D. Robb', 'New York, NY, USA'),
       ('J. K. Rowling', 'London, UK'),
       ('J. R. R. Tolkien', 'Bournmouth, UK'),
       ('Stephen King', 'Portland, ME, USA'),
       ('George R. R. Martin', 'Bayonne, NJ, USA'),
       ('Dan Brown', 'Exeter, NH, USA'),
       ('J. D. Salinger', 'New York, NY, USA'),
       ('J. R. R. Martin', 'Bayonne, NJ, USA'),
       ('J. D. Robb', 'New York, NY, USA');

insert into publisher (name, address)
values ('Bloomsbury', 'London, UK'),
       ('Allen & Unwin', 'Sydney, Australia'),
       ('Houghton Mifflin', 'Boston, MA, USA'),
       ('HarperCollins', 'New York, NY, USA'),
       ('Bloomsbury', 'London, UK'),
       ('Allen & Unwin', 'Sydney, Australia'),
       ('Houghton Mifflin', 'Boston, MA, USA'),
       ('HarperCollins', 'New York, NY, USA');

insert into seller (name)
values ('Amazon'),
       ('Barnes & Noble'),
       ('Books-A-Million'),
       ('Amazon'),
       ('Barnes & Noble'),
       ('Books-A-Million');

insert into book (title, author_id, publisher_id, year, price, seller_id)
values ('Harry Potter and the Philosopher''s Stone', 1, 1, 1997, 8.99, 1),
       ('Harry Potter and the Chamber of Secrets', 1, 1, 1998, 8.99, 1),
       ('Harry Potter and the Prisoner of Azkaban', 1, 1, 1999, 8.99, 1),
       ('Harry Potter and the Goblet of Fire', 1, 1, 2000, 8.99, 1),
       ('Harry Potter and the Order of the Phoenix', 1, 1, 2003, 8.99, 1),
       ('Harry Potter and the Half-Blood Prince', 1, 1, 2005, 8.99, 1),
       ('Harry Potter and the Deathly Hallows', 1, 1, 2007, 8.99, 1),
       ('The Fellowship of the Ring', 2, 2, 1954, 8.99, 2),
       ('The Two Towers', 2, 2, 1954, 8.99, 2),
       ('The Return of the King', 2, 2, 1955, 8.99, 2),
       ('The Shining', 3, 3, 1977, 8.99, 3),
       ('The Stand', 3, 3, 1978, 8.99, 3),
       ('It', 3, 3, 1986, 8.99, 3),
       ('The Gunslinger', 4, 4, 1982, 8.99, 4),
       ('The Drawing of the Three', 4, 4, 1987, 8.99, 4),
       ('The Waste Lands', 4, 4, 1991, 8.99, 4),
       ('The Dark Tower', 4, 4, 2003, 8.99, 4),
       ('The Da Vinci Code', 5, 5, 2003, 8.99, 5),
       ('Angels & Demons', 5, 5, 2000, 8.99, 5),
       ('The Lost Symbol', 5, 5, 2009, 8.99, 5),
       ('The Catcher in the Rye', 6, 6, 1951, 8.99, 6),
       ('The Hobbit', 7, 7, 1937, 8.99, 6),
       ('The Silmarillion', 7, 7, 1977, 8.99, 6),
       ('A Game of Thrones', 8, 8, 1996, 8.99, 6),
       ('A Clash of Kings', 8, 8, 1998, 8.99, 6),
       ('A Storm of Swords', 8, 8, 2000, 8.99, 6),
       ('A Feast for Crows', 8, 8, 2005, 8.99, 6),
       ('A Dance with Dragons', 8, 8, 2011, 8.99, 6);