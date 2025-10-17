#ifndef SORTJETS_H
#define SORTJETS_H

/*
  Sorts jets or jet constituents based on inference scores (highest to lowest).

  Example usage in RDataFrame:

  includePaths = ["functions.h", "SortJets.h"]

  df = df.Define("SortedJets",
                 "FCCAnalyses::JetClusteringUtils::JetSorter::sort_jets_by_score(Jets, JetScores)");

  df = df.Define("SortedJetConstituents",
                 "FCCAnalyses::JetClusteringUtils::JetSorter::sort_jetconstituents_by_score(JetConstituents, JetScores)");
*/

#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include "ROOT/RVec.hxx"

// If needed, include your FCCAnalysesJetConstituents definition
// #include "FCCAnalysesJetConstituents.h"

namespace FCCAnalyses { namespace JetClusteringUtils {

struct JetSorter {

  JetSorter() = default;

  // Sort fastjet::PseudoJets by descending score
  static ROOT::VecOps::RVec<fastjet::PseudoJet> sort_jets_by_score(
      const ROOT::VecOps::RVec<fastjet::PseudoJet> &jets,
      const ROOT::VecOps::RVec<float> &scores)
  {
    if (jets.size() != scores.size())
      throw std::runtime_error("JetSorter::sort_jets_by_score: jets and scores must have the same size");

    std::vector<size_t> indices(jets.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&](size_t i, size_t j) { return scores[i] > scores[j]; });

    ROOT::VecOps::RVec<fastjet::PseudoJet> sortedJets;
    sortedJets.reserve(jets.size());
    for (auto i : indices)
      sortedJets.push_back(jets[i]);

    return sortedJets;
  }

  // Sort FCCAnalysesJetConstituents by descending score
  static ROOT::VecOps::RVec<FCCAnalysesJetConstituents> sort_jetconstituents_by_score(
      const ROOT::VecOps::RVec<FCCAnalysesJetConstituents> &jcs,
      const ROOT::VecOps::RVec<float> &scores)
  {
    if (jcs.size() != scores.size())
      throw std::runtime_error("JetSorter::sort_jetconstituents_by_score: jet constituents and scores must have the same size");

    std::vector<size_t> indices(jcs.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&](size_t i, size_t j) { return scores[i] > scores[j]; });

    ROOT::VecOps::RVec<FCCAnalysesJetConstituents> sortedJCs;
    sortedJCs.reserve(jcs.size());
    for (auto i : indices)
      sortedJCs.push_back(jcs[i]);

    return sortedJCs;
  }
};

}} // namespace FCCAnalyses::JetClusteringUtils

#endif // SORTJETS_H
