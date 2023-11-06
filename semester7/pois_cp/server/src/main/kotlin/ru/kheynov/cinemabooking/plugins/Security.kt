package ru.kheynov.cinemabooking.plugins

import com.auth0.jwt.JWT
import com.auth0.jwt.algorithms.Algorithm
import io.ktor.server.application.*
import io.ktor.server.auth.*
import io.ktor.server.auth.jwt.*
import org.koin.ktor.ext.inject
import ru.kheynov.cinemabooking.jwt.token.TokenConfig

fun Application.configureSecurity() {
    val config: TokenConfig by inject()
    install(Authentication)
    authentication {
        jwt {
            verifier(
                JWT.require(Algorithm.HMAC256(config.secret)).withAudience(config.audience).withIssuer(config.issuer)
                    .build(),
            )
            validate { token ->
                if (token.payload.audience.contains(config.audience) && token.payload.expiresAt.time > System.currentTimeMillis()) {
                    JWTPrincipal(token.payload)
                } else {
                    null
                }
            }
        }
    }
}
