import scipp as sc


class Detector:
    'Description of the geometry of the Amor detector'
    # number of active blades in the detector
    nBlades = sc.scalar(14)
    # number of wires per blade
    nWires = sc.scalar(32)
    # number of stipes per blade
    nStripes = sc.scalar(64)
    # angle of incidence of the beam on the blades (def: 5.1)
    angle = sc.scalar(5.1, unit='degree').to(unit='rad')
    # height-distance of neighboring pixels on one blade
    dZ = sc.scalar(4.0, unit='mm') * sc.sin(angle)
    # depth-distance of neighboring pixels on one blade
    dX = sc.scalar(4.0, unit='mm') * sc.cos(angle)
    # distance between detector blades
    bladeZ = sc.scalar(10.455, unit='mm')
    # vertical center of the detector
    zero = 0.5 * nBlades.value * bladeZ
    # distance from focal point to leading blade edge
    distance = sc.scalar(4000, unit='mm')


def pixel_coordinate_in_detector_system(pixelID):
    """determine spatial coordinates and angles from pixel number"""
    pixelID.unit = ''
    (bladeNr, bPixel) = pixelID // (Detector.nWires * Detector.nStripes), pixelID % (
        Detector.nWires * Detector.nStripes
    )
    # z index on blade, y index on detector
    (bZi, detYi) = (
        bPixel // Detector.nStripes,
        bPixel % Detector.nStripes,
    )
    # z index on detector
    detZi = bladeNr * Detector.nWires + bZi
    # x position in detector
    detX = bZi * Detector.dX

    bladeAngle = (2.0 * sc.asin(0.5 * Detector.bladeZ / Detector.distance)).to(
        unit='degree'
    )
    delta = (Detector.nBlades / 2.0 - bladeNr) * bladeAngle - (
        sc.atan(bZi * Detector.dZ / (Detector.distance + bZi * Detector.dX))
    ).to(unit='degree')

    # z is in the direction of the center of the beam, y is the direction 'up'
    pixelID.unit = None
    return detYi, detZi, detX, delta


def pixel_coordinate_in_lab_frame(pixelID, nu):
    _, detZi, detX, delta = pixel_coordinate_in_detector_system(pixelID)

    angle_to_horizon = (nu + delta).to(unit='rad')
    distance_to_pixel = detX + Detector.distance

    # TODO: put the correct value here
    global_X = sc.zeros(dims=pixelID.dims, shape=pixelID.shape, unit='mm')
    global_Y = distance_to_pixel * sc.sin(angle_to_horizon)
    global_Z = distance_to_pixel * sc.cos(angle_to_horizon)
    return global_X, global_Y, global_Z
