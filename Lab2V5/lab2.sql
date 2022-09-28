drop database if exists lab2;
create database lab2;
\c lab2;

drop role if exists lab2_user;
create role lab2_user with login password '7P2bv6QIKEc';
alter database lab2 owner to lab2_user;

create type steering_side as enum ('LEFT', 'RIGHT');

create type type_of_drive as enum ('AWD', 'FWD', 'RWD');

create type body_type as enum ('Sedan', 'Wagon', 'Minibus');

create table if not exists characteristics
(
    id                  serial primary key,
    steering_side       steering_side not null,
    transmission        text          not null,
    fuel_injection_type text          not null,
    fuel_type           text          not null,
    type_of_drive       type_of_drive not null
);

create table if not exists brand
(
    id        serial primary key,
    name      text      not null,
    body_type body_type not null
);

create table if not exists car
(
    id                 serial primary key,
    name               text                                        not null,
    date_of_issue      date check ( date_of_issue < current_date ) not null,
    price              integer                                     not null,
    car_brand_id       integer references brand (id)               not null,
    characteristics_id integer references characteristics (id)     not null
);

-- Incorrect
insert into characteristics(steering_side, transmission, fuel_injection_type, fuel_type, type_of_drive)
values ('Right', 'Automatic', 'TSI', 'Gasoline', 'FWD');

insert into characteristics(steering_side, transmission, fuel_injection_type, fuel_type, type_of_drive)
values ('LEFT', 'Manual', 'TSI', 'Diesel', 'All Wheels Drive');

-- Correct
insert into characteristics(steering_side, transmission, fuel_injection_type, fuel_type, type_of_drive)
values ('RIGHT', 'Automatic', 'TSI', 'Gasoline', 'FWD');

insert into characteristics(steering_side, transmission, fuel_injection_type, fuel_type, type_of_drive)
values ('LEFT', 'Manual', 'TSI', 'Diesel', 'AWD');


-- Incorrect
insert into brand (name, body_type)
values ('Toyota', 'Hatchback');

-- Correct
insert into brand (name, body_type)
values ('Toyota', 'Sedan');

-- Correct
insert into car (name, date_of_issue, price, car_brand_id, characteristics_id)
values ('Vista', '10-12-2020', 4000, 1, 1);

-- Incorrect
insert into car (name, date_of_issue, price, car_brand_id, characteristics_id)
values ('Hilux', '22-12-2023', 400000, 1, 2);


alter table car alter column price set default 10;




begin;
alter table characteristics
    add column power integer;

savepoint alter_table;

create or replace view car_info as
select car.name                      as car_name,
       brand.body_type               as body_type,
       characteristics.steering_side as steering_side,
       characteristics.type_of_drive as type_of_drive,
       characteristics.fuel_type     as fuel_type,
       car.price                     as price

from car
         join brand on brand.id = car.car_brand_id
         join characteristics on car.characteristics_id = characteristics.id;

rollback to alter_table;

create or replace view car_info as
select car.name                      as car_name,
       brand.body_type               as body_type,
       characteristics.steering_side as steering_side,
       characteristics.type_of_drive as type_of_drive,
       characteristics.fuel_type     as fuel_type,
       car.price                     as price

from car
         join brand on brand.id = car.car_brand_id
         join characteristics on car.characteristics_id = characteristics.id;

commit;