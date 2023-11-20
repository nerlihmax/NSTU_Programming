package ru.kheynov.hotel.api.requests.users

import kotlinx.serialization.Serializable

@Serializable
data class UpdateUserRequest(
    val username: String,
)