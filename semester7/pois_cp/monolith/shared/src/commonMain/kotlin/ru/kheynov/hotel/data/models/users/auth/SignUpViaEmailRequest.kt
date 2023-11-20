package ru.kheynov.cinemabooking.api.requests.users.auth

import kotlinx.serialization.Serializable

@Serializable
data class SignUpViaEmailRequest(
    val username: String,
    val email: String,
    val password: String,
)