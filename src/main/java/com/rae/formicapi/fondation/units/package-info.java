/**
 * Physical unit configuration for client-side display.
 *
 * <p>Units in this package are used to format simulation quantities for
 * presentation. The simulation engine always operates on raw SI values
 * internally; these types only control how values are displayed to the user.
 *
 * <p>The active units are configured via {@code UnitConfig}, which exposes
 * one {@code ConfigEnum} per quantity type with a user-facing default.
 *
 * <h2>Configurable quantities</h2>
 * <ul>
 *   <li>{@link com.rae.formicapi.fondation.units.Temperature} — default °C</li>
 *   <li>{@link com.rae.formicapi.fondation.units.Pressure} — default atm</li>
 *   <li>{@link com.rae.formicapi.fondation.units.IrradiationFlux} — default Bq</li>
 * </ul>
 */
package com.rae.formicapi.fondation.units;