drop schema if exists public cascade;
create schema if not exists public;

create table client_firm
(
    id   serial primary key,
    name text not null
);

create table contract_type
(
    id   serial primary key,
    name text not null
);

create table contract
(
    id               serial primary key,
    client_firm_id   integer not null references client_firm (id),
    start_date       date    not null,
    validity         integer not null check ( validity > 0 ),
    contract_type_id integer not null references contract_type (id)
);


insert into client_firm (name)
values ('Client 1'),
       ('ИП Иванов Иван Иванович'),
       ('ООО "Рога и копыта'),
       ('ООО "Северный ветер"');

insert into contract_type (name)
values ('Договор Купли-продажи'),
       ('Договор поставки'),
       ('Договор аренды'),
       ('Договор подряда'),
       ('Договор оказания услуг'),
       ('Договор займа'),
       ('Договор комиссии'),
       ('Договор дарения'),
       ('Договор мены'),
       ('Договор ренты'),
       ('Договор хранения'),
       ('Договор страхования'),
       ('Договор поручения'),
       ('Договор перевозки'),
       ('Договор агентирования'),
       ('Договор субаренды'),
       ('Договор совместного управления'),
       ('Договор совместного владения имуществом');

insert into contract (client_firm_id, start_date, validity, contract_type_id)
values ((select id from client_firm order by random() limit 1), '2019-01-24', (random() * 730),
        (select id from contract_type order by random() limit 1)),
       ((select id from client_firm order by random() limit 1), '2020-04-11', (random() * 730),
        (select id from contract_type order by random() limit 1)),
       ((select id from client_firm order by random() limit 1), '2019-01-24', (random() * 730),
        (select id from contract_type order by random() limit 1)),
       ((select id from client_firm order by random() limit 1), '2020-04-11', (random() * 730),
        (select id from contract_type order by random() limit 1)),
       ((select id from client_firm order by random() limit 1), '2019-01-24', (random() * 730),
        (select id from contract_type order by random() limit 1)),
       ((select id from client_firm order by random() limit 1), '2020-04-11', (random() * 730),
        (select id from contract_type order by random() limit 1)),
       ((select id from client_firm order by random() limit 1), '2019-01-24', (random() * 730),
        (select id from contract_type order by random() limit 1)),
       ((select id from client_firm order by random() limit 1), '2020-04-11', (random() * 730),
        (select id from contract_type order by random() limit 1)),
       ((select id from client_firm order by random() limit 1), '2019-01-24', (random() * 730),
        (select id from contract_type order by random() limit 1)),
       ((select id from client_firm order by random() limit 1), '2020-04-11', (random() * 730),
        (select id from contract_type order by random() limit 1));


create table users
(
    id           serial primary key,
    login        text not null,
    password     text not null,
    access_level int  not null default 0
);

insert into users (login, password, access_level)
values ('user1', '5ad84jWgIVI', 2),
       ('user2', 'nE8EYarZ9fw', 1);

select contract.id, client_firm.name, contract_type.name, contract.start_date, contract.validity
from contract
         inner join client_firm on contract.client_firm_id = client_firm.id
         inner join contract_type on contract.contract_type_id = contract_type.id;

select contract.id,
       type.name as type,
       contract.start_date,
       contract.validity
from contract
         inner join contract_type type on type.id = contract.contract_type_id
where client_firm_id = 1;

select distinct client_firm.id,
                client_firm.name
from client_firm
         join contract on client_firm.id = contract.client_firm_id
where (select count(*) from contract where client_firm_id = id) > 0
order by id