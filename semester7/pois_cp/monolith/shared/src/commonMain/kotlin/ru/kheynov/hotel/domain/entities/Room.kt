package ru.kheynov.hotel.domain.entities

data class Room(
    val id: String,
    val type: String,
    val price: Int,
    val hotel: Hotel,
)
