package ru.kheynov.hotel.shared.domain.entities

import kotlinx.serialization.Serializable

@Serializable
data class Hotel(
    val id: Int,
    val name: String,
    val address: String,
    val city: String,
    val stars: Int,
)
