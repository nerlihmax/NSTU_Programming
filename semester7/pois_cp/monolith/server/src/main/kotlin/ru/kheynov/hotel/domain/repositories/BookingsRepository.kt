package ru.kheynov.hotel.domain.repositories

import ru.kheynov.hotel.domain.entities.Booking

interface BookingsRepository {
    suspend fun getBookingsByUserId(userId: String): List<String>
    suspend fun addBooking(userId: String, booking: Booking): Boolean
    suspend fun deleteBooking(userId: String, bookingId: String): Boolean
}