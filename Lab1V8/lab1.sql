DROP SCHEMA IF EXISTS v8 CASCADE;
CREATE SCHEMA IF NOT EXISTS v8;

CREATE TYPE v8.payment_state AS ENUM ('PAID', 'PARTIALLY PAID', 'UNPAID');

CREATE TABLE IF NOT EXISTS v8.cities
(
    id   SERIAL PRIMARY KEY,
    name text UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS v8.lodgers
(
    id          serial PRIMARY KEY,
    first_name  varchar(30) NOT NULL,
    second_name varchar(30) NOT NULL
);

CREATE TABLE IF NOT EXISTS v8.repairers
(
    id            serial PRIMARY KEY,
    first_name    varchar(30)                       NOT NULL,
    second_name   varchar(30)                       NOT NULL,
    date_of_birth date                              NOT NULL,
    city_id       integer REFERENCES v8.cities (id) NOT NULL
);

CREATE TABLE IF NOT EXISTS v8.repairments
(
    id                  serial PRIMARY KEY,
    type                text                                 NOT NULL,
    repairer_id         integer REFERENCES v8.repairers (id) NOT NULL,
    repairment_duration interval
);

CREATE TABLE IF NOT EXISTS v8.repair_facilities
(
    id              serial PRIMARY KEY,
    lodger_id       integer REFERENCES v8.lodgers (id),
    repairment_date date,
    price           integer NOT NULL,
    payment_state   v8.payment_state,
    repairment_id   integer REFERENCES v8.repairments (id)
);


INSERT INTO v8.lodgers (first_name, second_name)
VALUES ('Василий', 'Абобович'),
       ('Ильдар', 'Виноградов'),
       ('Лев', 'Калинин'),
       ('Белла', 'Капустина'),
       ('Дана', 'Соломина'),
       ('Стелла', 'Бородина'),
       ('Адам', 'Александров'),
       ('Ника', 'Воронина'),
       ('Карина', 'Ибрагимова'),
       ('Ахмет', 'Логинов')
ON CONFLICT DO NOTHING;

INSERT INTO v8.cities(name)
VALUES ('Нижний Новгород'),
       ('Москва'),
       ('Екатиринбург'),
       ('Санкт-Петербург'),
       ('Новосибирск'),
       ('Бердск'),
       ('Ульяновск'),
       ('Барнаул'),
       ('Владивосток'),
       ('Сочи'),
       ('Искитим')
ON CONFLICT DO NOTHING;

INSERT INTO v8.repairers (first_name, second_name, date_of_birth, city_id)
VALUES ('Виктор', 'Ватов', '1955-01-15', 1),
       ('Любовь', 'Муравьёва', '1981-07-28', 2),
       ('Наталья', 'Ларионова', '1962-12-15', 3),
       ('Зоя', 'Измайлова', '1973-02-24', 1),
       ('Карп', 'Воронин', '1993-12-04', 5),
       ('Пётр', 'Боголюбов', '1997-10-11', 6),
       ('Иван', 'Петров', '2000-06-05', 9),
       ('Сергей', 'Беляков', '1979-05-18', 5),
       ('Евгений', 'Крюков', '1977-09-11', 8),
       ('Дмитрий', 'Долгоруков', '1999-01-25', 4)
ON CONFLICT DO NOTHING;

INSERT INTO v8.repairments (type, repairer_id, repairment_duration)
VALUES ('Замена окон', 2, '1 week'),
       ('Окрашивание дверей', 1, '1 week'),
       ('Поклеивание обоев', 4, '3 hours'),
       ('Замена ламината', 6, '2 weeks'),
       ('Установка телевизора', 5, '1 hour'),
       ('Удаление осиного гнезда', 7, '2 hours'),
       ('Изоляция дверных проёмов', 8, '2 days'),
       ('Сборка шкафа-купе', 5, '5 hours'),
       ('Ремонт сантехники', 9, '1 hour'),
       ('Замена замка на входной двери', 1, '3 hours')
ON CONFLICT DO NOTHING;

INSERT INTO v8.repair_facilities (lodger_id, repairment_date, price, payment_state, repairment_id)
VALUES (1, '2020-01-22', 5000, 'PAID', 1),
       (2, '2021-02-23', 6000, 'PAID', 2),
       (4, '2017-03-12', 10000, 'PARTIALLY PAID', 3),
       (5, '2018-04-01', 2000, 'PAID', 4),
       (6, '2022-05-12', 2000, 'UNPAID', 5),
       (9, '2016-06-28', 3000, 'PAID', 6),
       (8, '2014-07-04', 5000, 'UNPAID', 7),
       (7, '2022-03-06', 15000, 'PAID', 8),
       (10, '2020-09-08', 1500, 'PARTIALLY PAID', 9),
       (2, '2012-11-22', 5000, 'PAID', 10)
ON CONFLICT DO NOTHING;


CREATE OR REPLACE VIEW v8.repairment_info AS
SELECT lodger.second_name             AS lodger_surname,
       repairment.type                AS repairment_type,
       repairer.second_name           AS repairer_surname,
       city.name                      AS city,
       repairment.repairment_duration AS repairment_duration,
       facility.repairment_date       AS repairment_date,
       facility.price                 AS price
FROM v8.repair_facilities AS facility
         INNER JOIN v8.lodgers AS lodger ON lodger.id = facility.lodger_id
         INNER JOIN v8.repairments AS repairment ON repairment.id = facility.repairment_id
         INNER JOIN v8.repairers AS repairer ON repairer.id = repairment.repairer_id
         INNER JOIN v8.cities AS city ON city.id = repairer.city_id;

SELECT *
FROM v8.repairment_info;