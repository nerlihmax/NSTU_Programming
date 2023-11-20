package ru.kheynov.hotel.data.models.users.auth

import kotlinx.serialization.Serializable

@Serializable
data class LoginViaEmailRequest(
    val email: String,
    val password: String,
)