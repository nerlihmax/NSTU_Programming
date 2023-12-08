package ru.kheynov.hotel.data.mappers

import ru.kheynov.hotel.data.entities.User
import ru.kheynov.hotel.shared.domain.entities.UserInfo
import ru.kheynov.hotel.shared.domain.entities.User as UserDomain

fun User.mapToUserInfo(): UserInfo = UserInfo(
    id = this.userId,
    name = this.name,
    email = this.email,
    passwordHash = this.passwordHash,
)

fun User.mapToUser(): UserDomain =
    UserDomain(
        id = this.userId,
        name = this.name,
        email = this.email,
    )