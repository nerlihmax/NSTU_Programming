package ru.kheynov.cinemabooking.data.mappers

import ru.kheynov.cinemabooking.data.entities.User
import ru.kheynov.cinemabooking.domain.entities.UserDTO

fun User.mapToUser(): UserDTO.User = UserDTO.User(
    userId = this.userId,
    username = this.name,
    email = this.email,
    passwordHash = this.passwordHash,
    authProvider = this.authProvider,
)