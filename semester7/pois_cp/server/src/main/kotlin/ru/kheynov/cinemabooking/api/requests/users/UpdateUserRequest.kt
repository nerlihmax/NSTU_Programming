package ru.kheynov.cinemabooking.api.requests.users

import kotlinx.serialization.Serializable

@Serializable
data class UpdateUserRequest(
    val username: String,
)