package ru.kheynov.cinemabooking.domain.repositories

import ru.kheynov.cinemabooking.domain.entities.Booking
import java.time.LocalDate

interface BookingsRepository {
    suspend fun getBookingsByUserId(userId: String): List<String>
    suspend fun addBooking(userId: String, booking: Booking): Boolean
    suspend fun deleteBooking(userId: String, bookingId: String): Boolean
}