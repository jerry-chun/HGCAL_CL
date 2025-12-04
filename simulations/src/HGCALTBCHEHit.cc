#include "HGCALTBCHEHit.hh"

HGCALTBCHEHit::HGCALTBCHEHit()
  : G4VHit(), fCHESignals({}), fX(-1.), fY(-1.), fZ(-1.), fEdep(0.),
    fTrackID(0), fShowerID(0), fLayer(-1), fPurity(0.0) {}

HGCALTBCHEHit::HGCALTBCHEHit(const HGCALTBCHEHit& right)
  : G4VHit(),
    fCHESignals(right.fCHESignals),
    fX(right.fX), fY(right.fY), fZ(right.fZ),
    fEdep(right.fEdep),
    fTrackID(right.fTrackID), fShowerID(right.fShowerID),
    fLayer(right.fLayer), fPurity(right.fPurity) {}

HGCALTBCHEHit::~HGCALTBCHEHit() = default;   

const HGCALTBCHEHit& HGCALTBCHEHit::operator=(const HGCALTBCHEHit& right)
{
  if (this == &right) return *this;
  fCHESignals = right.fCHESignals;
  fX = right.fX; fY = right.fY; fZ = right.fZ;
  fEdep = right.fEdep;
  fTrackID = right.fTrackID; fShowerID = right.fShowerID;
  fLayer = right.fLayer;
  fPurity    = right.fPurity;
  return *this;
}

G4bool HGCALTBCHEHit::operator==(const HGCALTBCHEHit& right) const
{
  return (this == &right);
}
