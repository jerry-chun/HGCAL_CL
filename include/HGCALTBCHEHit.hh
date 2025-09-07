#ifndef HGCALTBCHEHit_h
#define HGCALTBCHEHit_h 1

#include "HGCALTBConstants.hh"
#include "G4THitsCollection.hh"
#include "G4VHit.hh"
#include <array>

class HGCALTBCHEHit : public G4VHit
{
  public:
    HGCALTBCHEHit();
    HGCALTBCHEHit(const HGCALTBCHEHit&);
    virtual ~HGCALTBCHEHit();

    const HGCALTBCHEHit& operator=(const HGCALTBCHEHit&);
    G4bool operator==(const HGCALTBCHEHit&) const;

    virtual void Draw() {}
    virtual void Print() {}

    void SetPosition(G4double x, G4double y, G4double z) { fX = x; fY = y; fZ = z; }
    void SetEdep(G4double e) { fEdep = e; }

    G4double GetX() const { return fX; }
    G4double GetY() const { return fY; }
    G4double GetZ() const { return fZ; }
    G4double GetEdep() const { return fEdep; }

    void AddCellEdep(G4int CellID, G4double Edep);

    std::array<G4double, HGCALTBConstants::CHECells + 1> GetCHESignals() const
    {
      return fCHESignals;
    }

    void SetTrackID(G4int id) { fTrackID = id; }
    G4int GetTrackID() const { return fTrackID; }

    void SetShowerID(G4int id) { fShowerID = id; }
    G4int GetShowerID() const { return fShowerID; }
    
    void SetLayer(G4int l) { fLayer = l; }
    G4int GetLayer() const { return fLayer; }
    

  private:
    std::array<G4double, HGCALTBConstants::CHECells + 1> fCHESignals;
    G4double fX, fY, fZ;
    G4double fEdep;
    G4int fTrackID;
    G4int fShowerID;
    G4int fLayer = -1;
};

using HGCALTBCHEHitsCollection = G4THitsCollection<HGCALTBCHEHit>;

inline void HGCALTBCHEHit::AddCellEdep(G4int CellID, G4double Edep)
{
  fCHESignals[CellID] += Edep;
}

#endif
