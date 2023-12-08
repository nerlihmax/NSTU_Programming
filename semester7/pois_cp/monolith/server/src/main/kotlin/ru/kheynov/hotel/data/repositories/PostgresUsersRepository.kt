package ru.kheynov.hotel.data.repositories

import org.ktorm.database.Database
import org.ktorm.dsl.and
import org.ktorm.dsl.eq
import org.ktorm.dsl.from
import org.ktorm.dsl.limit
import org.ktorm.dsl.map
import org.ktorm.dsl.select
import org.ktorm.dsl.where
import org.ktorm.entity.add
import org.ktorm.entity.find
import org.ktorm.entity.sequenceOf
import ru.kheynov.hotel.data.entities.RefreshTokens
import ru.kheynov.hotel.data.entities.User
import ru.kheynov.hotel.data.entities.Users
import ru.kheynov.hotel.data.mappers.mapToUser
import ru.kheynov.hotel.data.mappers.mapToUserInfo
import ru.kheynov.hotel.data.mappers.toDataRefreshToken
import ru.kheynov.hotel.data.mappers.toRefreshTokenInfo
import ru.kheynov.hotel.shared.domain.entities.RefreshTokenInfo
import ru.kheynov.hotel.shared.domain.entities.UserInfo
import ru.kheynov.hotel.shared.domain.entities.UserUpdate
import ru.kheynov.hotel.shared.domain.repository.UsersRepository
import ru.kheynov.hotel.shared.jwt.RefreshToken
import ru.kheynov.hotel.shared.domain.entities.User as UserDomain

class PostgresUsersRepository(
    private val database: Database,
) : UsersRepository {
    override suspend fun registerUser(user: UserDomain, passwordHash: String): Boolean {
        val affectedRows = database.sequenceOf(Users).add(
            User {
                userId = user.id
                name = user.name
                email = user.email
                this.passwordHash = passwordHash
            },
        )
        return affectedRows == 1
    }

    override suspend fun getUserInfoByID(id: String): UserInfo? {
//        val clientIds =
//            database.from(RefreshTokens).selectDistinct(RefreshTokens.clientId)
//                .where { RefreshTokens.userId eq id }
//                .map { row -> row[RefreshTokens.clientId]!! }

        return database.from(Users).select(
            Users.userId,
            Users.name,
            Users.email,
            Users.passwordHash
        ).where(Users.userId eq id).limit(1).map { row ->
            UserInfo(
                id = row[Users.userId]!!,
                name = row[Users.name]!!,
                email = row[Users.email]!!,
                passwordHash = row[Users.passwordHash]!!
            )
        }.firstOrNull()
    }

    override suspend fun getUserByID(id: String): UserDomain? =
        database
            .sequenceOf(Users)
            .find { it.userId eq id }
            ?.mapToUser()

    override suspend fun deleteUserByID(id: String): Boolean {
        val affectedRows =
            database.sequenceOf(Users).find { user -> user.userId eq id }?.delete()
        return affectedRows == 1
    }

    override suspend fun updateUserByID(id: String, update: UserUpdate): Boolean {
        val foundUser = database.sequenceOf(Users).find { it.userId eq id } ?: return false
        if (update.name != null) foundUser.name = update.name!!

        val affectedRows = foundUser.flushChanges()
        return affectedRows == 1
    }

    override suspend fun getUserByEmail(email: String): UserInfo? {
        val foundUser = database.sequenceOf(Users).find { it.email eq email } ?: return null
        return foundUser.mapToUserInfo()
    }

    override suspend fun updateUserRefreshToken(
        userId: String,
        clientId: String,
        newRefreshToken: String,
        refreshTokenExpiration: Long,
    ): Boolean {
        val foundUser = database.sequenceOf(RefreshTokens)
            .find { (it.userId eq userId) and (it.clientId eq clientId) }
            ?: return false
        foundUser.refreshToken = newRefreshToken
        foundUser.expiresAt = refreshTokenExpiration
        val affectedRows = foundUser.flushChanges()
        return affectedRows == 1
    }

    override suspend fun getRefreshToken(oldRefreshToken: String): RefreshTokenInfo? {
        return database.sequenceOf(RefreshTokens).find { oldRefreshToken eq it.refreshToken }
            ?.toRefreshTokenInfo()
    }

    override suspend fun getRefreshToken(
        userId: String,
        clientId: String
    ): RefreshTokenInfo? {
        return database.sequenceOf(RefreshTokens)
            .find { (userId eq it.userId) and (clientId eq it.clientId) }
            ?.toRefreshTokenInfo()
    }

    override suspend fun createRefreshToken(
        userId: String,
        clientId: String,
        refreshToken: RefreshToken
    ): Boolean {
        val affectedRows = database.sequenceOf(RefreshTokens)
            .add(refreshToken.toDataRefreshToken(userId, clientId))
        return affectedRows == 1
    }
}