package ru.kheynov.cinemabooking.data.repositories

import org.ktorm.database.Database
import org.ktorm.dsl.*
import org.ktorm.entity.add
import org.ktorm.entity.find
import org.ktorm.entity.sequenceOf
import ru.kheynov.cinemabooking.data.entities.RefreshTokens
import ru.kheynov.cinemabooking.data.entities.User
import ru.kheynov.cinemabooking.data.entities.Users
import ru.kheynov.cinemabooking.data.mappers.mapToUser
import ru.kheynov.cinemabooking.data.mappers.toDataRefreshToken
import ru.kheynov.cinemabooking.data.mappers.toRefreshTokenInfo
import ru.kheynov.cinemabooking.domain.entities.UserDTO
import ru.kheynov.cinemabooking.domain.repositories.UsersRepository
import ru.kheynov.cinemabooking.jwt.token.RefreshToken

class PostgresUsersRepository(
    private val database: Database,
) : UsersRepository {
    override suspend fun registerUser(user: UserDTO.User): Boolean {
        val affectedRows = database.sequenceOf(Users).add(
            User {
                userId = user.userId
                name = user.username
                email = user.email
                passwordHash = user.passwordHash
                authProvider = user.authProvider
            },
        )
        return affectedRows == 1
    }

    override suspend fun getUserByID(userId: String): UserDTO.UserInfo? {
        val clientIds =
            database.from(RefreshTokens).selectDistinct(RefreshTokens.clientId).where { RefreshTokens.userId eq userId }
                .map { row -> row[RefreshTokens.clientId]!! }

        return database.from(Users).select(
                Users.userId,
                Users.name,
                Users.email,
                Users.authProvider,
            ).where(Users.userId eq userId).limit(1).map { row ->
                UserDTO.UserInfo(
                    userId = row[Users.userId]!!,
                    username = row[Users.name]!!,
                    email = row[Users.email]!!,
                    clientIds = clientIds,
                )
            }.firstOrNull()
    }

    override suspend fun deleteUserByID(userId: String): Boolean {
        val affectedRows = database.sequenceOf(Users).find { user -> user.userId eq userId }?.delete()
        return affectedRows == 1
    }

    override suspend fun updateUserByID(userId: String, update: UserDTO.UpdateUser): Boolean {
        val foundUser = database.sequenceOf(Users).find { it.userId eq userId } ?: return false

        if (update.username != null) foundUser.name = update.username

        val affectedRows = foundUser.flushChanges()
        return affectedRows == 1
    }

    override suspend fun getUserByEmail(email: String): UserDTO.User? {
        val foundUser = database.sequenceOf(Users).find { it.email eq email } ?: return null
        return foundUser.mapToUser()
    }

    override suspend fun updateUserRefreshToken(
        userId: String,
        clientId: String,
        newRefreshToken: String,
        refreshTokenExpiration: Long,
    ): Boolean {
        val foundUser = database.sequenceOf(RefreshTokens).find { (it.userId eq userId) and (it.clientId eq clientId) }
            ?: return false
        foundUser.refreshToken = newRefreshToken
        foundUser.expiresAt = refreshTokenExpiration
        val affectedRows = foundUser.flushChanges()
        return affectedRows == 1
    }

    override suspend fun getRefreshToken(oldRefreshToken: String): UserDTO.RefreshTokenInfo? {
        return database.sequenceOf(RefreshTokens).find { oldRefreshToken eq it.refreshToken }?.toRefreshTokenInfo()
    }

    override suspend fun getRefreshToken(userId: String, clientId: String): UserDTO.RefreshTokenInfo? {
        return database.sequenceOf(RefreshTokens).find { (userId eq it.userId) and (clientId eq it.clientId) }
            ?.toRefreshTokenInfo()
    }

    override suspend fun createRefreshToken(userId: String, clientId: String, refreshToken: RefreshToken): Boolean {
        val affectedRows = database.sequenceOf(RefreshTokens).add(refreshToken.toDataRefreshToken(userId, clientId))
        return affectedRows == 1
    }
}