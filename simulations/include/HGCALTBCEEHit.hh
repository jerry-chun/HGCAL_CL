#ifndef HGCALTBCEEHit_h
#define HGCALTBCEEHit_h 1

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4ThreeVector.hh"

class HGCALTBCEEHit : public G4VHit
{
  public:
    HGCALTBCEEHit();
    HGCALTBCEEHit(const HGCALTBCEEHit&);
    virtual ~HGCALTBCEEHit();

    const HGCALTBCEEHit& operator=(const HGCALTBCEEHit&);
    G4bool operator==(const HGCALTBCEEHit&) const;

    virtual void Draw() {}
    virtual void Print() {}

    void SetPosition(G4double x, G4double y, G4double z);
    G4double GetX() const { return fX; }
    G4double GetY() const { return fY; }
    G4double GetZ() const { return fZ; }

    void SetEdep(G4double edep) { fEdep = edep; }
    G4double GetEdep() const { return fEdep; }

    void SetTrackID(G4int id) { fTrackID = id; }
    G4int GetTrackID() const { return fTrackID; }

    void SetShowerID(G4int id) { fShowerID = id; }
    G4int GetShowerID() const { return fShowerID; }
    
    void SetLayer(G4int l) { fLayer = l; }
    G4int GetLayer() const { return fLayer; }

  private:
    G4double fX, fY, fZ;
    G4double fEdep;
    G4int fTrackID;
    G4int fShowerID;
    G4int fLayer = -1;
};

using HGCALTBCEEHitsCollection = G4THitsCollection<HGCALTBCEEHit>;

#endif
