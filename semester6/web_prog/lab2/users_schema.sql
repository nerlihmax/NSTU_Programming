create table users
(
    id           serial primary key,
    login        text not null,
    password     text not null,
    access_level int  not null default 0
);

insert into users (login, password, access_level)
values ('admin', 'qwertyuiop', 2),
       ('user', '1234567890', 1);