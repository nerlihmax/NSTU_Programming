package ru.kheynov.hotel.data.mappers

import ru.kheynov.hotel.data.entities.User
import ru.kheynov.hotel.domain.entities.UserDTO

fun User.mapToUser(): UserDTO.User = UserDTO.User(
    userId = this.userId,
    username = this.name,
    email = this.email,
    passwordHash = this.passwordHash,
    authProvider = this.authProvider,
)