
#include "HGCALTBCEEHit.hh"

HGCALTBCEEHit::HGCALTBCEEHit()
  : G4VHit(), fX(-1.), fY(-1.), fZ(-1.), fEdep(0.), fPurity(0.0) {}

HGCALTBCEEHit::HGCALTBCEEHit(const HGCALTBCEEHit& right)
  : G4VHit(), fX(right.fX), fY(right.fY), fZ(right.fZ), fEdep(right.fEdep), fPurity(right.fPurity) {}

HGCALTBCEEHit::~HGCALTBCEEHit() {}

const HGCALTBCEEHit& HGCALTBCEEHit::operator=(const HGCALTBCEEHit& right)
{
  fX = right.fX;
  fY = right.fY;
  fZ = right.fZ;
  fEdep = right.fEdep;
  fPurity   = right.fPurity;
  return *this;
}

G4bool HGCALTBCEEHit::operator==(const HGCALTBCEEHit& right) const
{
  return (this == &right);
}

void HGCALTBCEEHit::SetPosition(G4double x, G4double y, G4double z)
{
  fX = x;
  fY = y;
  fZ = z;
}
