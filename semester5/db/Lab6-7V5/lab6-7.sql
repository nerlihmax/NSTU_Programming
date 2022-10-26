drop schema if exists l67 cascade;
create schema if not exists l67;

create type l67.steering_side as enum ('LEFT', 'RIGHT');

create type l67.type_of_drive as enum ('AWD', 'FWD', 'RWD');

create type l67.body_type as enum ('Sedan', 'Wagon', 'Minibus', 'Hatchback');

create table if not exists l67.characteristics
(
    id                  serial primary key,
    steering_side       l67.steering_side not null,
    transmission        text              not null,
    fuel_injection_type text              not null,
    fuel_type           text              not null,
    type_of_drive       l67.type_of_drive not null,
    engine_volume       integer
);

create table if not exists l67.countries
(
    country_code char(2) primary key unique,
    name         text not null
);

create table if not exists l67.brand
(
    id           serial primary key,
    name         text          not null,
    body_type    l67.body_type not null,
    country_code char(2) references l67.countries (country_code)
);

create table if not exists l67.manufacturer
(
    id   serial primary key not null,
    name text               not null
);

create table if not exists l67.car
(
    id                 serial primary key,
    name               text                                        not null,
    date_of_issue      date check ( date_of_issue < current_date ) not null,
    price              integer                                     not null,
    car_brand_id       integer references l67.brand (id),
    manufacturer_id    integer references l67.manufacturer (id),
    characteristics_id integer references l67.characteristics (id) not null
);

create table if not exists l67.showroom
(
    id      serial primary key,
    name    text not null,
    address text not null
);

create table if not exists l67.seller
(
    id          serial primary key,
    first_name  text                                 not null,
    second_name text,
    showroom_id integer references l67.showroom (id) not null
);

create table if not exists l67.selling
(
    id          serial primary key,
    car_id      integer references l67.car (id)    not null,
    seller_id   integer references l67.seller (id) not null,
    showroom_id integer references l67.showroom (id),
    date        date
);

create table if not exists l67.available_cars
(
    id          serial primary key,
    car_id      integer references l67.car (id)      not null,
    showroom_id integer references l67.showroom (id) not null
);

insert into l67.countries
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
       ('BZ', 'Brazil'),
       ('CA', 'Canada')
on conflict do nothing;


insert into l67.manufacturer(name)
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
       ('Chevrolet, US'),
       ('AVTO, China'),
       ('CAR, Canada'),
       ('AVTO, Canada')
on conflict do nothing;

insert into l67.brand(name, country_code, body_type)
values ('Toyota', 'JP', 'Sedan'),
       ('BMW', 'GE', 'Sedan'),
       ('Honda', 'JP', 'Wagon'),
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
       ('Chevrolet', 'US', 'Sedan'),
       ('Toyota', 'JP', 'Wagon'),
       ('CAR', 'CA', 'Wagon'),
       ('AVTO', 'CN', 'Wagon'),
       ('AVTO', 'CA', 'Minibus')
on conflict do nothing;

insert into l67.characteristics(steering_side, transmission, fuel_injection_type, fuel_type, type_of_drive,
                                engine_volume)
values ('RIGHT', 'Automatic', 'TSI', 'Gasoline', 'FWD', 2200),
       ('LEFT', 'Manual', 'KE-Jetronic', 'Gasoline', 'RWD', 2800),
       ('LEFT', 'CVT', 'MPI', 'Gasoline', 'FWD', 3000),
       ('LEFT', 'Automatic', 'TSI', 'Diesel', 'FWD', 3000),
       ('LEFT', 'Robotic', 'D4', 'Gasoline', 'FWD', 4800),
       ('LEFT', 'Manual', 'MPI', 'Gasoline', 'AWD', 1600),
       ('LEFT', 'Automatic', 'None', 'Electric', 'RWD', null),
       ('LEFT', 'CVT', 'TSI', 'Gasoline', 'FWD', '2500'),
       ('LEFT', 'Automatic', 'None', 'Electric', 'FWD', null),
       ('LEFT', 'Robotic', 'TSI', 'Diesel', 'AWD', 2000),
       ('RIGHT', 'Automatic', 'MPI', 'Gasoline', 'FWD', 1100)
on conflict do nothing;

insert into l67.car(name, manufacturer_id, date_of_issue, price, car_brand_id, characteristics_id)
values ('Vitz', 1, '11-12-2002', 400000, 1, 1),
       ('520D', 2, '19-11-2021', 4000000, 2, 4),
       ('Accord', 3, '01-10-2008', 800000, 3, 3),
       ('Discovery 3', 4, '05-06-2005', 1000000, 4, 4),
       ('Discovery 3', 5, '13-09-2006', 1100000, 4, 4),
       ('XF', 6, '16-05-2011', 800000, 5, 1),
       ('Vista', 1, '11-06-1994', 215000, 1, 1),
       ('Corolla', 1, '11-06-1994', 480000, 18, 1),
       ('Corolla', 1, '11-06-1994', 480000, 1, 11),
       ('Corolla', 1, '11-06-1994', 480000, 1, 1),
       ('Sprinter', 1, '11-06-1994', 140000, 18, 11),
       ('S500', 7, '13-09-2010', 2000000, 6, 2),
       ('Vesta', 8, '06-11-2018', 1300000, 7, 6),
       ('Urus', 9, '17-02-2022', 40000000, 8, 10),
       ('Solaris', 10, '11-04-2022', 1600000, 9, 6),
       ('Creta', 11, '24-02-2022', 2100000, 9, 6),
       ('Rio', 12, '03-03-2021', 1600000, 10, 6),
       ('Mohave', 13, '06-01-2022', 4000000, 10, 8),
       ('C4', 15, '11-09-2013', 800000, 11, 2),
       ('C4', 15, '11-09-2013', 800000, 11, 2),
       ('Nexia', 16, '05-01-2011', 200000, 12, 6),
       ('Polo', 18, '16-04-2022', 1400000, 13, 6),
       ('Mondeo', 20, '24-01-2010', 400000, 14, 6),
       ('F7', 22, '07-01-2022', 2100000, 15, 8),
       ('Granta', 8, '10-01-2019', 800000, 7, 6),
       ('Camaro', 24, '12-10-2021', 2500000, 17, 2),
       ('Corolla', 1, '11-06-2020', 500000, 1, 11),
       ('Camry', 1, '11-06-2022', 1600000, 1, 6),
       ('Land Cruiser', 1, '11-06-2020', 4800000, 1, 4),
       ('C4', 15, '11-09-2022', 4000000, 11, 2),
       ('Odyssey', 3, '01-01-2018', 950000, 3, 3),
       ('Accord', 3, '01-01-2018', 950000, 3, 3),
       ('Stream', 3, '01-01-2018', 950000, 3, 3),
       ('Wizard', 25, '11-06-2022', 1600000, 20, 6),
       ('Fleet', 26, '11-06-2022', 1600000, 19, 6),
       ('Screamer', 27, '11-06-2020', 1800000, 21, 6)
on conflict do nothing;

insert into l67.showroom(name, address)
values ('АЦ Фрунзе', 'Фрунзе, 252'),
       ('Восток моторс', 'Большевистская, 276/2'),
       ('АЦ Сибирский тракт', 'Жуковского, 96/2'),
       ('Эксперт НСК', 'Большевистская, 276/1')
on conflict do nothing;

insert into l67.seller(first_name, second_name, showroom_id)
values ('Viktor', 'Vatov', 1),
       ('Dmitry', 'Pelevin', 2),
       ('Ilya', 'Amogusov', 3),
       ('Ivan', 'Kalita', 4)
on conflict do nothing;

insert into l67.available_cars(car_id, showroom_id)
values (1, 1),
       (2, 1),
       (3, 1),
       (4, 2),
       (5, 2),
       (6, 2),
       (7, 3),
       (8, 3),
       (9, 3),
       (10, 4),
       (11, 4),
       (12, 4),
       (28, 2),
       (29, 2),
       (30, 3),
       (31, 2),
       (32, 2),
       (33, 2),
       (34, 1),
       (35, 4),
       (36, 1)
on conflict do nothing;

insert into l67.selling(car_id, seller_id, showroom_id, date)
values (13, 1, 1, '2020-10-11'),
       (14, 2, 2, '2019-09-01'),
       (15, 3, 3, '2018-08-07'),
       (16, 4, 4, '2017-07-06'),
       (17, 2, 1, '2022-06-12'),
       (18, 3, 2, '2011-05-14'),
       (19, 1, 3, '2015-04-21'),
       (20, 4, 4, '2016-03-22'),
       (21, 1, 1, '2021-02-24'),
       (22, 2, 2, '2020-01-16'),
       (23, 3, 4, '2018-12-18'),
       (24, 1, 3, '2016-11-05'),
       (25, 4, 2, '2020-09-14'),
       (26, 2, 1, '2021-10-18'),
       (27, 4, 3, '2022-01-19')
on conflict do nothing;

create or replace function add_auto(model_id int, market_id int, num int) returns char(30)
as
$$
declare
    count integer;
begin

end;
$$
    language 'plpgsql';
