package ru.kheynov.hotel.shared.domain.entities

import kotlinx.serialization.Serializable

@Serializable
data class UserEmployment(
    val isEmployee: Boolean,
    val hotel: Hotel? = null,
)
