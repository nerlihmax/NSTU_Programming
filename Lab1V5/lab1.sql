drop schema if exists v5 cascade;
create schema if not exists v5;

create type v5.steering_side as enum ('LEFT', 'RIGHT');

create table if not exists v5.characteristics
(
    id                  serial primary key,
    steering_side       v5.steering_side not null,
    transmission        text             not null,
    fuel_injection_type text             not null,
    fuel_type           text             not null,
    type_of_drive       text             not null
);

alter table v5.characteristics
    add column power int;

create table if not exists v5.manufacturer
(
    id   serial primary key not null,
    name text               not null
);

create table if not exists v5.countries
(
    country_code char(2) primary key unique,
    name         text not null
);

create table if not exists v5.brand
(
    id           serial primary key,
    name         text                                           not null,
    country_code char(2) references v5.countries (country_code) not null,
    body_type    text                                           not null
);

create table if not exists v5.car
(
    id                 serial primary key,
    name               text                                       not null,
    manufacturer_id    integer references v5.manufacturer (id)    not null,
    year_of_issue      integer                                    not null,
    price              integer                                    not null,
    car_brand_id       integer references v5.brand (id)           not null,
    characteristics_id integer references v5.characteristics (id) not null
);


insert into v5.countries
values ('RU', 'Russia'),
       ('JP', 'Japan'),
       ('GB', 'United Kingdom'),
       ('CN', 'China'),
       ('US', 'United States'),
       ('GE', 'Germany'),
       ('KR', 'South Korea'),
       ('UZ', 'Uzbekistan'),
       ('FR', 'France'),
       ('IT', 'Italy'),
       ('BZ', 'Brazil')
on conflict do nothing;


insert into v5.manufacturer(name)
values ('Toyota In Some Japanese city idk im not japanese'),
       ('BMW, Kaliningrad, Russia'),
       ('Honda, Tokyo'),
       ('Land Rover, India'),
       ('Land Rover, UK'),
       ('Jaguar, UK'),
       ('Mercedes-Benz, Novgorod'),
       ('Lada, Tolyatti'),
       ('Lamborghini, Italy'),
       ('Hyundai, Russia'),
       ('Hyundai, Korea'),
       ('Hyundai, US'),
       ('Kia, US'),
       ('Kia, Korea'),
       ('Citroen, France'),
       ('Daewoo, Uzbekistan'),
       ('Volkswagen, Brazil'),
       ('Volkswagen, Germany'),
       ('Volkswagen, Russia'),
       ('Ford, US'),
       ('Ford, India'),
       ('Haval, China'),
       ('Haval, Russia')
on conflict do nothing;

insert into v5.brand(name, country_code, body_type)
values ('Toyota', 'JP', 'Sedan'),
       ('BMW', 'GE', 'Hatchback'),
       ('Honda', 'JP', 'Sedan'),
       ('Land Rover', 'GB', 'SUV'),
       ('Jaguar', 'GB', 'Sedan'),
       ('Mercedes-Benz', 'GE', 'Sedan'),
       ('Lada', 'RU', 'Sedan'),
       ('Lamborghini', 'IT', 'SUV'),
       ('Hyundai', 'KR', 'SUV'),
       ('Kia', 'KR', 'Sedan'),
       ('Citroen', 'FR', 'Sedan'),
       ('Daewoo', 'UZ', 'Sedan'),
       ('Volkswagen', 'GE', 'Hatchback'),
       ('Ford', 'US', 'SUV'),
       ('Haval', 'CN', 'SUV')
on conflict do nothing;

insert into v5.characteristics(steering_side, transmission, fuel_injection_type, fuel_type, type_of_drive)
values ('RIGHT', 'Automatic', 'TSI', 'Gasoline', 'FWD'),
       ('LEFT', 'Manual', 'KE-Jetronic', 'Gasoline', 'RWD'),
       ('LEFT', 'CVT', 'MPI', 'Gasoline', 'FWD'),
       ('LEFT', 'Automatic', 'TSI', 'Diesel', 'FWD'),
       ('LEFT', 'Robotic', 'D4', 'Gasoline', 'FWD'),
       ('LEFT', 'Manual', 'MPI', 'Diesel', 'AWD'),
       ('LEFT', 'Automatic', 'None', 'Electric', 'RWD'),
       ('LEFT', 'CVT', 'TSI', 'Gasoline', 'FWD'),
       ('LEFT', 'Automatic', 'None', 'Electric', 'FWD'),
       ('LEFT', 'Robotic', 'TSI', 'Diesel', 'AWD')
on conflict do nothing;

insert into v5.car(name, manufacturer_id, year_of_issue, price, car_brand_id, characteristics_id)
values ('Vitz', 1, 2002, 400000, 1, 1),
       ('520D', 2, 2021, 4000000, 2, 4),
       ('Accord', 3, 2008, 800000, 3, 3),
       ('Discovery 3', 4, 2005, 1000000, 4, 4),
       ('Discovery 3', 5, 2006, 1100000, 4, 4),
       ('XF', 6, 2011, 800000, 5, 1),
       ('S500', 7, 2010, 2000000, 6, 2),
       ('Vesta', 8, 2018, 1300000, 7, 6),
       ('Urus', 9, 2022, 40000000, 8, 10),
       ('Solaris', 10, 2022, 1600000, 9, 6),
       ('Creta', 11, 2022, 2100000, 9, 6),
       ('Rio', 12, 2021, 1600000, 10, 6),
       ('Mohave', 13, 2020, 4000000, 10, 8),
       ('C4', 15, 2013, 400000, 11, 2),
       ('Nexia', 16, 2011, 200000, 12, 6),
       ('Polo', 18, 2022, 1400000, 13, 6),
       ('Mondeo', 20, 2012, 400000, 14, 6),
       ('F7', 22, 2022, 2100000, 15, 8)
on conflict do nothing;

create or replace view v5.car_info as
select v5.car.name                      as car_name,
       v5.countries.name                as vendor_homeland,
       v5.manufacturer.name             as manufacturer_country,
       v5.brand.body_type               as body_type,
       v5.characteristics.steering_side as steering_side,
       v5.characteristics.type_of_drive as type_of_drive,
       v5.characteristics.fuel_type     as fuel_type,
       v5.car.price                     as price

from v5.car
         inner join v5.brand on v5.brand.id = v5.car.car_brand_id
         inner join v5.manufacturer on v5.manufacturer.id = v5.car.manufacturer_id
         inner join v5.characteristics on v5.car.characteristics_id = characteristics.id
         inner join v5.countries on v5.brand.country_code = v5.countries.country_code;

select *
from v5.car_info;