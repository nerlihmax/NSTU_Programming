drop schema if exists public cascade;
create schema if not exists public;

create table countries
(
    id           serial primary key,
    country_code char(2) unique not null,
    name         text
);

create table providers
(
    id         serial primary key,
    name       text,
    country_id integer references countries (id)
);

create table products
(
    id             serial primary key,
    name           text,
    date_of_issue  date,
    country_id     integer references countries (id),
    date_of_supply date,
    is_defect      boolean,
    provider_id    integer references providers (id),
    quantity       integer,
    price          integer
);


create table sales
(
    id         serial primary key,
    product_id integer references products (id),
    quantity   integer
);

grant all privileges on all tables in schema public to admin;

revoke all privileges on all tables in schema public from operator;
revoke all privileges on all tables in schema public from db_user;

grant insert, select, update, delete on countries, providers to operator;

grant insert, select, update, delete on products, sales to db_user;
