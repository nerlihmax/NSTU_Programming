package ru.kheynov.hotel.shared.domain.mappers

import ru.kheynov.hotel.shared.domain.entities.User
import ru.kheynov.hotel.shared.domain.entities.UserInfo

fun UserInfo.toUser(): User = User(
    id = this.id,
    name = this.name,
    email = this.email,
)