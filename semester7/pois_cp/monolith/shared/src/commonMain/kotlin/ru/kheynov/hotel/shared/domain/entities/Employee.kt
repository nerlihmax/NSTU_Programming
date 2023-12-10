package ru.kheynov.hotel.shared.domain.entities

import kotlinx.serialization.Serializable

@Serializable
data class Employee(
    val id: Int,
    val user: User,
    val hotel: Hotel,
)
