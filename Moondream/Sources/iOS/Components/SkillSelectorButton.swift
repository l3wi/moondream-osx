import SwiftUI
import MoondreamKit

#if os(iOS)
/// Skill selector button with Liquid Glass morphing animation
/// The circular icon morphs into the selected skill row at the bottom of the expanded list
/// Expands upward and rightward with bottom-left corner anchored
/// Uses overlay so expanded state doesn't shift other UI elements
struct SkillSelectorButton: View {
    @Binding var selectedSkill: Skill
    @State private var isExpanded = false
    @Namespace private var skillNamespace

    /// Skills ordered with selected at bottom for proper morphing
    private var orderedSkills: [Skill] {
        let others = Skill.allCases.filter { $0 != selectedSkill }
        return others + [selectedSkill]
    }

    var body: some View {
        // Fixed size container - expanded menu is overlaid, doesn't affect parent layout
        ZStack(alignment: .bottomLeading) {
            // Invisible spacer to maintain consistent size for parent layout
            Color.clear
                .frame(width: 56, height: 56)

            // The actual content (collapsed button or expanded menu)
            content
        }
    }

    @ViewBuilder
    private var content: some View {
        // Container spacing controls morphing - views within this distance will blend
        GlassEffectContainer(spacing: 20) {
            if isExpanded {
                // Expanded state: full list with all skills
                VStack(spacing: 0) {
                    // Other skills (above selected)
                    ForEach(orderedSkills.dropLast()) { skill in
                        SkillOptionRow(
                            skill: skill,
                            isSelected: false
                        ) {
                            withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                                selectedSkill = skill
                                isExpanded = false
                            }
                        }

                        Divider()
                            .background(Color.white.opacity(0.2))
                    }

                    // Selected skill at bottom (tap to collapse)
                    Button {
                        withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                            isExpanded = false
                        }
                    } label: {
                        HStack(spacing: 12) {
                            Image(systemName: selectedSkill.icon)
                                .font(.title3)
                                .foregroundStyle(.white)
                                .frame(width: 28)

                            VStack(alignment: .leading, spacing: 2) {
                                Text(selectedSkill.displayName)
                                    .font(.subheadline)
                                    .fontWeight(.medium)
                                    .foregroundStyle(.white)

                                Text(selectedSkill.description)
                                    .font(.caption2)
                                    .foregroundStyle(.white.opacity(0.6))
                                    .lineLimit(1)
                            }

                            Spacer()

                            Image(systemName: "checkmark")
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .foregroundStyle(.white)
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
                .padding(.vertical, 8)
                .frame(width: 240)
                .glassEffect(.regular, in: .rect(cornerRadius: 16))
                .glassEffectID("skillSelector", in: skillNamespace)
                .glassEffectTransition(.matchedGeometry)
            } else {
                // Collapsed state: just the icon in a circle
                Button {
                    withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                        isExpanded = true
                    }
                } label: {
                    Image(systemName: selectedSkill.icon)
                        .font(.title2)
                        .foregroundStyle(.white)
                        .frame(width: 56, height: 56)
                }
                .glassEffect(.regular.interactive(), in: .circle)
                .glassEffectID("skillSelector", in: skillNamespace)
                .glassEffectTransition(.matchedGeometry)
            }
        }
    }
}

/// Individual skill option row in the expanded selector
struct SkillOptionRow: View {
    let skill: Skill
    let isSelected: Bool
    let onSelect: () -> Void

    var body: some View {
        Button(action: onSelect) {
            HStack(spacing: 12) {
                // Icon
                Image(systemName: skill.icon)
                    .font(.title3)
                    .foregroundStyle(isSelected ? .white : .white.opacity(0.8))
                    .frame(width: 28)

                // Title and subtitle
                VStack(alignment: .leading, spacing: 2) {
                    Text(skill.displayName)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundStyle(.white)

                    Text(skill.description)
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.6))
                        .lineLimit(1)
                }

                Spacer()

                // Selection indicator
                if isSelected {
                    Image(systemName: "checkmark")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                        .foregroundStyle(.white)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    ZStack {
        Color.black.ignoresSafeArea()

        VStack {
            Spacer()
            HStack {
                SkillSelectorButton(selectedSkill: .constant(.caption))
                Spacer()
                Circle()
                    .fill(.white.opacity(0.3))
                    .frame(width: 70, height: 70)
                Spacer()
                Circle()
                    .fill(.white.opacity(0.3))
                    .frame(width: 56, height: 56)
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 100)
        }
    }
}
#endif // os(iOS)
