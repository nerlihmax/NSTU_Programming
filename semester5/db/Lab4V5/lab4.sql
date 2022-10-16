drop schema if exists l4 cascade;
create schema if not exists l4;

create type l4.steering_side as enum ('LEFT', 'RIGHT');

create type l4.type_of_drive as enum ('AWD', 'FWD', 'RWD');

create type l4.body_type as enum ('Sedan', 'Wagon', 'Minibus', 'Hatchback');

create table if not exists l4.characteristics
(
    id                  serial primary key,
    steering_side       l4.steering_side not null,
    transmission        text             not null,
    fuel_injection_type text             not null,
    fuel_type           text             not null,
    type_of_drive       l4.type_of_drive not null,
    engine_volume       integer
);

create table if not exists l4.countries
(
    country_code char(2) primary key unique,
    name         text not null
);

create table if not exists l4.brand
(
    id           serial primary key,
    name         text         not null,
    body_type    l4.body_type not null,
    country_code char(2) references l4.countries (country_code)
);

create table if not exists l4.manufacturer
(
    id   serial primary key not null,
    name text               not null
);

create table if not exists l4.car
(
    id                 serial primary key,
    name               text                                        not null,
    date_of_issue      date check ( date_of_issue < current_date ) not null,
    price              integer                                     not null,
    car_brand_id       integer references l4.brand (id),
    manufacturer_id    integer references l4.manufacturer (id),
    characteristics_id integer references l4.characteristics (id)  not null
);

insert into l4.countries
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


insert into l4.manufacturer(name)
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
       ('Haval, Russia'),
       ('Chevrolet, US')
on conflict do nothing;

insert into l4.brand(name, country_code, body_type)
values ('Toyota', 'JP', 'Sedan'),
       ('BMW', 'GE', 'Sedan'),
       ('Honda', 'JP', 'Sedan'),
       ('Land Rover', 'GB', 'Minibus'),
       ('Jaguar', 'GB', 'Sedan'),
       ('Mercedes-Benz', 'GE', 'Sedan'),
       ('Lada', 'RU', 'Sedan'),
       ('Lamborghini', 'IT', 'Minibus'),
       ('Hyundai', 'KR', 'Minibus'),
       ('Kia', 'KR', 'Sedan'),
       ('Citroen', 'FR', 'Sedan'),
       ('Daewoo', 'UZ', 'Sedan'),
       ('Volkswagen', 'GE', 'Hatchback'),
       ('Ford', 'US', 'Minibus'),
       ('Haval', 'CN', 'Minibus'),
       ('Lada', 'RU', 'Sedan'),
       ('Chevrolet', 'US', 'Sedan')
on conflict do nothing;

insert into l4.characteristics(steering_side, transmission, fuel_injection_type, fuel_type, type_of_drive,
                               engine_volume)
values ('RIGHT', 'Automatic', 'TSI', 'Gasoline', 'FWD', '2200'),
       ('LEFT', 'Manual', 'KE-Jetronic', 'Gasoline', 'RWD', '2800'),
       ('LEFT', 'CVT', 'MPI', 'Gasoline', 'FWD', '3000'),
       ('LEFT', 'Automatic', 'TSI', 'Diesel', 'FWD', '3000'),
       ('LEFT', 'Robotic', 'D4', 'Gasoline', 'FWD', '4800'),
       ('LEFT', 'Manual', 'MPI', 'Gasoline', 'AWD', '1600'),
       ('LEFT', 'Automatic', 'None', 'Electric', 'RWD', null),
       ('LEFT', 'CVT', 'TSI', 'Gasoline', 'FWD', '2500'),
       ('LEFT', 'Automatic', 'None', 'Electric', 'FWD', null),
       ('LEFT', 'Robotic', 'TSI', 'Diesel', 'AWD', '2000')
on conflict do nothing;

insert into l4.car(name, manufacturer_id, date_of_issue, price, car_brand_id, characteristics_id)
values ('Vitz', 1, '2002-12-11', 400000, 1, 1),
       ('520D', 2, '2021-11-19', 4000000, 2, 4),
       ('Accord', 3, '2008-10-01', 800000, 3, 3),
       ('Discovery 3', 4, '2005-06-05', 1000000, 4, 4),
       ('Discovery 3', 5, '2006-09-13', 1100000, 4, 4),
       ('XF', 6, '2011-05-16', 800000, 5, 1),
       ('Vista', 1, '1994-06-11', 215000, 1, 1),
       ('Corolla', 1, '1994-06-11', 480000, 1, 1),
       ('Corolla', 1, '1994-06-11', 480000, 1, 1),
       ('Corolla', 1, '1994-06-11', 480000, 1, 1),
       ('Sprinter', 1, '1994-06-11', 140000, 1, 1),
       ('S500', 7, '2010-09-13', 2000000, 6, 2),
       ('Vesta', 8, '2018-11-06', 1300000, 7, 6),
       ('Urus', 9, '2022-02-17', 40000000, 8, 10),
       ('Solaris', 10, '2022-04-11', 1600000, 9, 6),
       ('Creta', 11, '2022-02-24', 2100000, 9, 6),
       ('Rio', 12, '2021-03-03', 1600000, 10, 6),
       ('Mohave', 13, '2022-01-06', 4000000, 10, 8),
       ('C4', 15, '2013-09-11', 800000, 11, 2),
       ('C4', 15, '2013-09-11', 800000, 11, 2),
       ('Nexia', 16, '2011-01-05', 200000, 12, 6),
       ('Polo', 18, '2022-04-16', 1400000, 13, 6),
       ('Mondeo', 20, '2010-01-24', 400000, 14, 6),
       ('F7', 22, '2022-01-07', 2100000, 15, 8),
       ('Granta', 8, '2019-01-10', 800000, 7, 6),
       ('Camaro', 24, '2021-10-12', 2500000, 17, 2),
       ('Corolla', 1, '2020-06-11', 900000, 1, 1),
       ('Camry', 1, '2021-06-11', 1600000, 1, 1),
       ('Land Cruiser', 1, '2020-06-11', 4800000, 1, 4),
       ('C4', 15, '2022-09-11', 4000000, 11, 2)
on conflict do nothing;

-- Select right-handed automatic transmission which are more expensive then average japanese mini cars
select brand.name as brand,
       car.name   as model,
       car.price,
       ch.type_of_drive,
       ch.transmission
from l4.car as car
         inner join l4.characteristics as ch on ch.id = car.characteristics_id
         inner join l4.brand as brand on brand.id = car.car_brand_id
where ch.transmission = 'Automatic'
  and ch.steering_side = 'RIGHT'
  and car.price > (select(avg(car.price))
                   from l4.car as car
                            inner join l4.brand brand on brand.id = car.car_brand_id
                            inner join l4.characteristics ch on ch.id = car.characteristics_id
                   where ch.engine_volume < 3000
                     and brand.country_code = 'JP');

-- Select German and US sedans and minibuses which are more expensive than cars manufactured in japan since last 3 years
select brand.name         as brand,
       brand.country_code as country,
       car.name           as model,
       car.price,
       ch.type_of_drive,
       ch.transmission
from l4.car as car
         inner join l4.characteristics as ch on ch.id = car.characteristics_id
         inner join l4.brand as brand on brand.id = car.car_brand_id
where (brand.country_code = 'GE' or brand.country_code = 'US')
  and (brand.body_type = 'Sedan' or brand.body_type = 'Minibus')
  and car.price > (select(avg(car.price))
                   from l4.car as car
                            inner join l4.brand brand on brand.id = car.car_brand_id
                   where brand.country_code = 'JP'
                     and car.date_of_issue > current_date::date - interval '3 years');


-- Select left-handed japanese and french cars more expensive than average manual russian cars
select brand.name         as brand,
       brand.country_code as country,
       car.name           as model,
       car.price,
       ch.type_of_drive,
       ch.transmission
from l4.car as car
         inner join l4.characteristics as ch on ch.id = car.characteristics_id
         inner join l4.brand as brand on brand.id = car.car_brand_id
where (brand.country_code = 'JP' or brand.country_code = 'FR')
  and car.price > (select(avg(car.price))
                   from l4.car as car
                            inner join l4.brand brand on brand.id = car.car_brand_id
                            inner join l4.characteristics as ch on ch.id = car.characteristics_id
                   where brand.country_code = 'RU'
                     and ch.transmission = 'Manual');