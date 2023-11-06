create table users
(
    user_id       text primary key,
    name          text,
    email         text not null,
    password_hash text,
    auth_provider text not null default 'local',
    address       text
);

create table refresh_tokens
(
    id         serial primary key,
    user_id    text   not null references users (user_id) on delete cascade,
    client_id  text   not null,
    token      text   not null unique,
    expires_at bigint not null
);
