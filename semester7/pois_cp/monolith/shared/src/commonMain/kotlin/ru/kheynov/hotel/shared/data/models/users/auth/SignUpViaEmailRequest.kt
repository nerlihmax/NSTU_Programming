package ru.kheynov.hotel.shared.data.models.users.auth

import kotlinx.serialization.Serializable

@Serializable
data class SignUpViaEmailRequest(
    val username: String,
    val email: String,
    val password: String,
)