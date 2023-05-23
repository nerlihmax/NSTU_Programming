drop schema if exists public cascade;
create schema public;

create table departments
(
    id   serial primary key,
    name text not null
);

create table teachers
(
    id         serial primary key,
    full_name  text not null,
    department int references departments (id),
    post       text not null,
    hire_date  date not null
);

create table groups
(
    id        serial primary key,
    name      text unique not null,
    specialty text        not null
);

create table disciplines
(
    id        serial primary key,
    name      text not null,
    semester  int check ( semester > 0 and semester <= 10 ),
    specialty text
);

create table disciplines_schedule
(
    id         serial primary key,
    discipline int references disciplines (id),
    teacher    int references teachers (id),
    hours      int check ( hours > 2 )
);

insert into departments (name)
values ('Кафедра ВТ'),
       ('Техническая поддержка'),
       ('Кафедра физики'),
       ('Кафедра гуманитарного образования');

insert into teachers (full_name, department, post, hire_date)
values ('Токарев Вадим Геннадьевич', 1, 'Профессор', '2019-01-28'),
       ('Быков Игорь Валерьевич', 2, 'Профессор', '2019-01-28'),
       ('Пейсахович Юрий Григорьевич', 3, 'Профессор', '2019-01-28'),
       ('Данилкова Марина Петровна', 4, 'Профессор', '2019-01-28');

insert into groups (name, specialty)
values ('АВТ-019', '09.03.01'),
       ('АВТ-119', '09.03.01'),
       ('АА-26', '27.03.04'),
       ('АА-17', '27.03.04'),
       ('АВТ-018', '09.03.01');

insert into disciplines (name, semester, specialty)
values ('Клиент-серверные приложения', 6, '09.03.01'),
       ('Физика', 2, '27.03.04'),
       ('Системное администрирование', 6, '09.03.01'),
       ('Философия', 3, '27.03.04');

insert into disciplines_schedule (discipline, teacher, hours)
values (1, 1, 72),
       (2, 3, 46),
       (3, 2, 18),
       (3, 1, 18),
       (4, 4, 48);

-- выдавать справки по преподавателю - какие дисциплины он читает, для каких групп и сколько часов
select t.full_name, d.name as discipline, g.name as group_name, ds.hours
from teachers t
         join disciplines_schedule ds on t.id = ds.teacher
         join disciplines d on ds.discipline = d.id
         join groups g on d.specialty = g.specialty
where t.id = 1;

--по запросу, указав код дисциплины, получить список преподавателей, ее читающих.
select t.full_name
from teachers t
         join disciplines_schedule ds on t.id = ds.teacher
where ds.discipline = 3;
