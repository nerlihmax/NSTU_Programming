package ru.kheynov.cinemabooking.domain.repositories

interface BookingsRepository {
    suspend fun getBookingsByUserId(userId: String): List<String>
    suspend fun addBooking(userId: String, bookingId: String): Boolean
    suspend fun deleteBooking(userId: String, bookingId: String): Boolean
}