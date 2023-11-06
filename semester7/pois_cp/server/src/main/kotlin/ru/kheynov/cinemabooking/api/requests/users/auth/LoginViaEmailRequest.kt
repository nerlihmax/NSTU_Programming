package ru.kheynov.cinemabooking.api.requests.users.auth

import kotlinx.serialization.Serializable

@Serializable
data class LoginViaEmailRequest(
    val email: String,
    val password: String,
)