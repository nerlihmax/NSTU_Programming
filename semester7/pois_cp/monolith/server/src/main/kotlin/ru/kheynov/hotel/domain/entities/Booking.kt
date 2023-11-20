package ru.kheynov.cinemabooking.domain.entities

import kotlinx.serialization.Serializable

@Serializable
data class Booking(
    val userId: String,
    val timetableId: String,
)