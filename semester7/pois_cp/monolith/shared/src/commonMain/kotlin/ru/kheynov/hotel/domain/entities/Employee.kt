package ru.kheynov.hotel.domain.entities

data class Employee(
    val id: Int,
    val user: User,
    val hotel: Hotel,
)
