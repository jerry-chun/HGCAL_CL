#ifndef HGCALTBEventAction_h
#define HGCALTBEventAction_h 1

#include <vector>
#include <string>
#include "G4UserEventAction.hh"
#include "globals.hh"
#include "HGCALTBAHHit.hh"
#include "HGCALTBCEEHit.hh"
#include "HGCALTBCHEHit.hh"
#include "HGCALTBConstants.hh"
#include "HGCALTBPrimaryGenAction.hh"
enum HGCALDetID : int { kCEE = 0, kCHE = 1, kAHCAL = 2 };
class HGCALTBEventAction : public G4UserEventAction
{
public:
    HGCALTBEventAction(HGCALTBPrimaryGenAction* PGA);
    HGCALTBEventAction();
    virtual ~HGCALTBEventAction();

    virtual void BeginOfEventAction(const G4Event* event);
    virtual void EndOfEventAction(const G4Event* event);

    void Addedep(G4double stepedep);
    void SetIntLayer(G4int IntTrack);

    std::vector<G4double>& GetCEESignals() { return fCEELayerSignals; };
    std::vector<G4double>& GetCHESignals() { return fCHELayerSignals; };
    std::vector<G4double>& GetAHCALSignals() { return fAHCALLayerSignals; };

    std::vector<G4double> hits_x, hits_y, hits_z, hits_Edep;
    std::vector<G4int> hit_layer;
    std::vector<G4int> hit_trackid;
    std::vector<G4int> hit_showerid;
    std::vector<G4int> hit_detector;
    std::vector<G4double>& GetPrimaryEnergies() { return fPrimaryGenAction->GetPrimaryEnergies(); }
    std::vector<G4int>&    GetPrimaryPDGIDs()   { return fPrimaryGenAction->GetPrimaryPDGIDs(); }
    
    G4int cee_hit_count;

private:
    HGCALTBCEEHitsCollection* GetCEEHitsCollection(G4int hcID, const G4Event* event) const;
    HGCALTBCHEHitsCollection* GetCHEHitsCollection(G4int hcID, const G4Event* event) const;
    HGCALTBAHCALHitsCollection* GetAHCALHitsCollection(G4int hcID, const G4Event* event) const;

    G4int fCEEHCID = -1;
    G4int fCHEHCID = -1;
    G4int fAHCALHCID = -1;
    G4double edep;
    G4int fIntLayer;

    std::vector<G4double> fCEELayerSignals;
    std::vector<G4double> fCHELayerSignals;
    std::vector<G4double> fAHCALLayerSignals;

    HGCALTBPrimaryGenAction* fPrimaryGenAction;
};

inline void HGCALTBEventAction::Addedep(G4double stepedep) { edep += stepedep; }
inline void HGCALTBEventAction::SetIntLayer(G4int IntTrack) { fIntLayer = IntTrack; }

#endif
