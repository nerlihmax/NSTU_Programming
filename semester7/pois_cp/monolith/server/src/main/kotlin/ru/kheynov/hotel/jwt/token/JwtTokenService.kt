package ru.kheynov.hotel.jwt.token

import com.auth0.jwt.JWT
import com.auth0.jwt.algorithms.Algorithm
import ru.kheynov.hotel.jwt.RefreshToken
import ru.kheynov.hotel.jwt.TokenPair
import java.util.Date
import java.util.UUID

class JwtTokenService : TokenService {
    override fun generateTokenPair(config: TokenConfig, vararg claims: TokenClaim): TokenPair {
        var token = JWT.create()
            .withAudience(config.audience)
            .withIssuer(config.issuer)
            .withExpiresAt(Date(System.currentTimeMillis() + config.accessLifetime * 1000))
        claims.forEach { claim ->
            token = token.withClaim(claim.name, claim.value)
        }
        val accessToken = token.sign(Algorithm.HMAC256(config.secret))
        val refreshToken = UUID.randomUUID().toString()
        val refreshTokenExpiration = System.currentTimeMillis() + config.refreshLifetime * 1000
        return TokenPair(accessToken, RefreshToken(refreshToken, refreshTokenExpiration))
    }
}