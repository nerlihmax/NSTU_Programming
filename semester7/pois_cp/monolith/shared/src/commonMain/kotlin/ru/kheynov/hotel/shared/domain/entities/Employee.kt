package ru.kheynov.hotel.shared.domain.entities

data class Employee(
    val id: Int,
    val user: User,
    val hotel: Hotel,
)
