package ru.kheynov.hotel.data.mappers

import ru.kheynov.hotel.data.entities.User
import ru.kheynov.hotel.domain.entities.UserInfo

fun User.mapToUserInfo(): UserInfo = UserInfo(
    id = this.userId,
    name = this.name,
    email = this.email,
    passwordHash = this.passwordHash,
)