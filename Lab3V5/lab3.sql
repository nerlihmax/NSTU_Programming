drop schema if exists l3 cascade;
create schema if not exists l3;

create type l3.steering_side as enum ('LEFT', 'RIGHT');

create type l3.type_of_drive as enum ('AWD', 'FWD', 'RWD');

create type l3.body_type as enum ('Sedan', 'Wagon', 'Minibus', 'Hatchback');

create table if not exists l3.characteristics
(
    id                  serial primary key,
    steering_side       l3.steering_side not null,
    transmission        text             not null,
    fuel_injection_type text             not null,
    fuel_type           text             not null,
    type_of_drive       l3.type_of_drive not null
);

create table if not exists l3.countries
(
    country_code char(2) primary key unique,
    name         text not null
);

create table if not exists l3.brand
(
    id           serial primary key,
    name         text         not null,
    body_type    l3.body_type not null,
    country_code char(2) references l3.countries (country_code)
);

create table if not exists l3.manufacturer
(
    id   serial primary key not null,
    name text               not null
);

create table if not exists l3.car
(
    id                 serial primary key,
    name               text                                        not null,
    date_of_issue      date check ( date_of_issue < current_date ) not null,
    price              integer                                     not null,
    car_brand_id       integer references l3.brand (id),
    manufacturer_id    integer references l3.manufacturer (id),
    characteristics_id integer references l3.characteristics (id)  not null
);

insert into l3.countries
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


insert into l3.manufacturer(name)
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

insert into l3.brand(name, country_code, body_type)
values ('Toyota', 'JP', 'Sedan'),
       ('BMW', 'GE', 'Hatchback'),
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
       ('Lada', 'RU', 'Sedan')
on conflict do nothing;

insert into l3.characteristics(steering_side, transmission, fuel_injection_type, fuel_type, type_of_drive)
values ('RIGHT', 'Automatic', 'TSI', 'Gasoline', 'FWD'),
       ('LEFT', 'Manual', 'KE-Jetronic', 'Gasoline', 'RWD'),
       ('LEFT', 'CVT', 'MPI', 'Gasoline', 'FWD'),
       ('LEFT', 'Automatic', 'TSI', 'Diesel', 'FWD'),
       ('LEFT', 'Robotic', 'D4', 'Gasoline', 'FWD'),
       ('LEFT', 'Manual', 'MPI', 'Gasoline', 'AWD'),
       ('LEFT', 'Automatic', 'None', 'Electric', 'RWD'),
       ('LEFT', 'CVT', 'TSI', 'Gasoline', 'FWD'),
       ('LEFT', 'Automatic', 'None', 'Electric', 'FWD'),
       ('LEFT', 'Robotic', 'TSI', 'Diesel', 'AWD')
on conflict do nothing;

insert into l3.car(name, manufacturer_id, date_of_issue, price, car_brand_id, characteristics_id)
values ('Vitz', 1, '2002-12-11', 400000, 1, 1),
       ('520D', 2, '2021-11-19', 4000000, 2, 4),
       ('Accord', 3, '2008-10-01', 800000, 3, 3),
       ('Discovery 3', 4, '2005-06-05', 1000000, 4, 4),
       ('Discovery 3', 5, '2006-09-13', 1100000, 4, 4),
       ('XF', 6, '2011-05-16', 800000, 5, 1),
       ('Vista', 1, '1994-06-11', 215000, 1, 1),
       ('Corolla', 1, '1994-06-11', 480000, 1, 1),
       ('Sprinter', 1, '1994-06-11', 140000, 1, 1),
       ('S500', 7, '2010-09-13', 2000000, 6, 2),
       ('Vesta', 8, '2018-11-06', 1300000, 7, 6),
       ('Urus', 9, '2022-02-17', 40000000, 8, 10),
       ('Solaris', 10, '2022-04-11', 1600000, 9, 6),
       ('Creta', 11, '2022-02-24', 2100000, 9, 6),
       ('Rio', 12, '2021-03-03', 1600000, 10, 6),
       ('Mohave', 13, '2022-01-06', 4000000, 10, 8),
       ('C4', 15, '2013-09-11', 400000, 11, 2),
       ('Nexia', 16, '2011-01-05', 200000, 12, 6),
       ('Polo', 18, '2022-04-16', 1400000, 13, 6),
       ('Mondeo', 20, '2010-01-24', 400000, 14, 6),
       ('F7', 22, '2022-01-07', 2100000, 15, 8),
       ('Granta', 8, '2019-01-10', 800000, 7, 6)
on conflict do nothing;

-- Select cheapest right-steering car
select l3.brand.name,
       l3.car.name,
       l3.car.date_of_issue,
       l3.car.price,
       l3.characteristics.steering_side,
       l3.characteristics.transmission,
       l3.brand.body_type
from l3.car
         join l3.characteristics on l3.car.characteristics_id = l3.characteristics.id
         join l3.brand on l3.car.car_brand_id = l3.brand.id
where l3.characteristics.steering_side = 'RIGHT'
order by price
limit 1;

-- Select the most expensive automatic car
select l3.brand.name,
       l3.car.name,
       l3.car.date_of_issue,
       l3.car.price,
       l3.characteristics.steering_side,
       l3.characteristics.transmission,
       l3.brand.body_type
from l3.car
         join l3.characteristics on l3.car.characteristics_id = l3.characteristics.id
         join l3.brand on l3.car.car_brand_id = l3.brand.id
where l3.characteristics.transmission = 'Automatic'
order by price desc
limit 1;

-- Select average FWD cars price
select avg(l3.car.price) as average_price
from l3.car
         join l3.characteristics c on c.id = l3.car.characteristics_id
where c.type_of_drive = 'FWD';

-- Select count of cars with body type sedan
select count(*) as sedans_count
from l3.car
         join l3.brand b on b.id = l3.car.car_brand_id
where b.body_type = 'Sedan';

-- Select total lada price
select sum(car.price) as total_lada_price
from l3.car as car
         join l3.brand as brand on brand.id = car.car_brand_id
where brand.name = 'Lada';


-- Common part

drop table if exists cars;
create table cars
(
    id    serial primary key,
    wheel text[] not null
);

insert into cars(wheel)
values ('{"wheel1", "wheel2", "wheel3", "wheel4", "steering wheel", "spare wheel"}'),
       ('{"wheel12", "wheel22", "wheel32", "wheel42", "steering wheel2", "spare wheel2"}'),
       ('{"wheel13", "wheel23", "wheel33", "wheel43", "steering wheel3", "spare wheel3"}'),
       ('{"wheel14", "wheel24", "wheel34", "wheel44", "steering wheel4", "spare wheel4"}'),
       ('{"wheel15", "wheel25", "wheel35", "wheel45", "steering wheel5", "spare wheel5"}')
on conflict do nothing;

select wheel[6]
from cars
where wheel[6] is not null;

select wheel[7]
from cars
where wheel[7] is not null;

select wheel[2:4]
from cars
where wheel[2:4] is not null;


drop table if exists matrices;

create table matrices
(
    id     serial primary key,
    matrix integer[][]
);

insert into matrices(matrix)
values ('{{1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}}'),
       ('{{2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5}, {2, 6}}'),
       ('{{3, 1}, {3, 2}, {3, 3}, {3, 4}, {3, 5}, {3, 6}}'),
       ('{{4, 1}, {4, 2}, {4, 3}, {4, 4}, {4, 5}, {4, 6}}'),
       ('{{5, 1}, {5, 2}, {5, 3}, {5, 4}, {5, 5}, {5, 6}}'),
       ('{{6, 1}, {6, 2}, {6, 3}, {6, 4}, {6, 5}, {6, 6}}')
on conflict do nothing;

select matrix[2][1]
from matrices
where matrix[2][1] is not null;

select matrix[2:4][2:3]
from matrices;


select array_dims(wheel)
from cars;

select array_dims(matrix)
from matrices;

update cars
set wheel[2]='{"wheel_updated"}'
where id = 1;

update matrices
set matrix[1:4]= '{{9, 9}, {9, 9}, {9, 9}, {9,9}}'
where id = 1