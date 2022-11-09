drop schema if exists public cascade;
create schema if not exists public;

drop role if exists operator;
drop role if exists db_user;

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

create role operator with login password 'qwerty';
create role db_user with login password 'qwerty';

grant usage on schema public to operator;
grant usage on schema public to db_user;
grant usage on schema public to admin;

grant all on schema public to admin;
grant all on schema public to operator;
grant all on schema public to db_user;

grant select, usage on all sequences in schema public to operator;
grant select, usage on all sequences in schema public to db_user;
grant all privileges on all sequences in schema public to admin;

grant insert, select, update, delete on countries, providers to operator;
grant insert, select, update, delete on products, sales to db_user;
grant all privileges on all tables in schema public to admin;
