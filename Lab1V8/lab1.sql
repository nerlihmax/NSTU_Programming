drop schema if exists v8 cascade;
create schema v8;

create type v8.payment_state as ENUM ('PAID', 'PARTIALLY PAID', 'UNPAID');

create table if not exists v8.cities
(
    id   serial primary key,
    name text unique not null
);

create table if not exists v8.repairers
(
    id            serial primary key,
    first_name    varchar(10)                       not null,
    second_name   varchar(10)                       not null,
    date_of_birth date                              not null,
    city_id       integer references v8.cities (id) not null
);

create table if not exists v8.repairments
(
    id                  serial primary key,
    type                text                                 not null,
    repairer_id         integer references v8.repairers (id) not null,
    repairment_duration interval
);

create table if not exists v8.repair_facilities
(
    id              serial primary key,
    repairment_date date,
    price           integer not null,
    payment_state   payment_state
);
