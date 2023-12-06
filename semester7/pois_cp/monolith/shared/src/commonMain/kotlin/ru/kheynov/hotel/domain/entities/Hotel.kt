package ru.kheynov.hotel.domain.entities

data class Hotel(
    val id: Int,
    val name: String,
    val address: String,
    val city: String,
    val stars: Int,
)
